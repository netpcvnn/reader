from __future__ import absolute_import
from __future__ import division

import time
import os
import math
import logging
import sys

import distance
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from PIL import Image
from six.moves import xrange
from six import BytesIO
from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from util.data_gen import DataGen
from util.visualizations import visualize_attention


class Model(object):
    def __init__(self,
                 phase,
                 visualize,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 steps_per_checkpoint,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 use_distance=True,
                 max_image_width=160,
                 max_image_height=60,
                 max_prediction_length=8,
                 channels=1,
                 reg_val=0):

        self.use_distance = use_distance

        # We need resized width, not the actual width
        self.max_original_width = max_image_width
        self.max_width = int(math.ceil(1. * max_image_width / max_image_height * DataGen.IMAGE_HEIGHT))

        self.encoder_size = int(math.ceil(1. * self.max_width / 4))
        self.decoder_size = max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        if gpu_id >= 0:
            device_id = '/gpu:' + str(gpu_id)
        else:
            device_id = '/cpu:0'
        self.device_id = device_id

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if use_gru:
            logging.info('using GRU in the decoder.')

        self.reg_val = reg_val
        self.sess = session
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
        self.channels = channels

        # if phase == 'train':
        #     self.forward_only = False
        # else:
        #     self.forward_only = True

        # if phase == 'train':
        # self.forward_only = False
        # else:
        self.forward_only = True



        with tf.device(device_id):

            self.height = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.int32)
            # print("height = ", self.height)
            self.height_float = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)

            self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')
            self.img_data = tf.cond(
                tf.less(tf.rank(self.img_pl), 1),
                lambda: tf.expand_dims(self.img_pl, 0),
                lambda: self.img_pl
            )

            # print(self.img_data)

            #convert to input tensor
            self.img_data = tf.map_fn(self._prepare_image, self.img_data, dtype=tf.float32)
            num_images = tf.shape(self.img_data)[0]

            # TODO: create a mask depending on the image/batch size
            self.encoder_masks = []
            for i in xrange(self.encoder_size + 1):
                self.encoder_masks.append(
                    tf.tile([[1.]], [num_images, 1])
                )

            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(self.decoder_size + 1):
                self.decoder_inputs.append(
                    tf.tile([0], [num_images])
                )
                if i < self.decoder_size:
                    self.target_weights.append(tf.tile([1.], [num_images]))
                else:
                    self.target_weights.append(tf.tile([0.], [num_images]))

            cnn_model = CNN(self.img_data, not self.forward_only)
            self.conv_output = cnn_model.tf_output()
            self.perm_conv_output = tf.transpose(self.conv_output, perm=[1, 0, 2])
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks=self.encoder_masks,
                encoder_inputs_tensor=self.perm_conv_output,
                decoder_inputs=self.decoder_inputs,
                target_weights=self.target_weights,
                target_vocab_size=len(DataGen.CHARMAP),
                buckets=self.buckets,
                target_embedding_size=target_embedding_size,
                attn_num_layers=attn_num_layers,
                attn_num_hidden=attn_num_hidden,
                forward_only=self.forward_only,
                use_gru=use_gru)

            table = tf.contrib.lookup.MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value="",
                checkpoint=True,
            )

            insert = table.insert(
                tf.constant(list(range(len(DataGen.CHARMAP))), dtype=tf.int64),
                tf.constant(DataGen.CHARMAP),
            )

            with tf.control_dependencies([insert]):
                num_feed = []
                prb_feed = []
                # print("len(self.attention_decoder_model.output) = ", len(self.attention_decoder_model.output)) = 102
                for l in xrange(len(self.attention_decoder_model.output)):
                    guess = tf.argmax(self.attention_decoder_model.output[l], axis=1)
                    # print("guess", guess)
                    proba = tf.reduce_max(
                        tf.nn.softmax(self.attention_decoder_model.output[l]), axis=1)
                    # print("proba", proba)
                    num_feed.append(guess)
                    prb_feed.append(proba)

                # Join the predictions into a single output string.
                trans_output = tf.transpose(num_feed)
                trans_output = tf.map_fn(
                    lambda m: tf.foldr(
                        lambda a, x: tf.cond(
                            tf.equal(x, DataGen.EOS_ID),
                            lambda: '',
                            lambda: table.lookup(x) + a
                        ),
                        m,
                        initializer=''
                    ),
                    trans_output,
                    dtype=tf.string
                )

                # Calculate the total probability of the output string.
                trans_outprb = tf.transpose(prb_feed)
                trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
                trans_outprb = tf.map_fn(
                    lambda m: tf.foldr(
                        lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                        m,
                        initializer=tf.cast(1, tf.float64)
                    ),
                    trans_outprb,
                    dtype=tf.float64
                )

                self.prediction = tf.cond(
                    tf.equal(tf.shape(trans_output)[0], 1),
                    lambda: trans_output[0],
                    lambda: trans_output,
                )
                self.probability = tf.cond(
                    tf.equal(tf.shape(trans_outprb)[0], 1),
                    lambda: trans_outprb[0],
                    lambda: trans_outprb,
                )

                self.prediction = tf.identity(self.prediction, name='prediction')
                self.probability = tf.identity(self.probability, name='probability')

            # if not self.forward_only:  # train
            self.updates = []
            self.summaries_by_bucket = []

            params = tf.trainable_variables()
            opt = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)

            if self.reg_val > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                logging.info('Adding %s regularization losses', len(reg_losses))
                logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                loss_op = self.reg_val * tf.reduce_sum(reg_losses) + self.attention_decoder_model.loss
            else:
                loss_op = self.attention_decoder_model.loss

            gradients, params = zip(*opt.compute_gradients(loss_op, params))
            if self.clip_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
            summaries = []
            summaries.append(tf.summary.scalar("loss", loss_op))
            summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
            
            
            summaries.append(tf.summary.scalar("validation", tf.global_norm(gradients)))
                        
            
            all_summaries = tf.summary.merge(summaries)
            self.summaries_by_bucket.append(all_summaries)
            # update op - apply gradients
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.updates.append(opt.apply_gradients(zip(gradients, params), global_step=self.global_step))


        self.saver_all = tf.train.Saver(tf.all_variables())
        self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print("model_checkpoint_path = ", ckpt.model_checkpoint_path) #/home/hoangtienduc/DoB/checkpoints/model.ckpt-30788
        if ckpt and load_model:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

    def predict(self, image_file_data):
        text, prob = self.sess.run([self.prediction, self.probability],
                                   feed_dict={self.img_pl.name: image_file_data})

        if sys.version_info >= (3,):
            text = text.decode('utf-8')

        return (text, prob)

    def batch_predict(self, data_path):
        s_gen = DataGen(data_path, self.buckets, epochs=1,
                        max_width=self.max_original_width)
        for batch in s_gen.gen(self.batch_size):
            result = self.step(batch, self.forward_only)

            preds = result['prediction']
            probs = result['probability']
            paths = batch['paths']
            for i in range(len(preds)):
                pred = preds[i].decode('utf-8')
                path = paths[i].decode('utf-8')
                img_name = path.split('/')[-1]
                prob = probs[i]

                print('{}\t{}\t{}'.format(img_name, pred, prob))

    def test(self, data_path):
        current_step = 0
        num_correct = 0.0
        num_total = 0.0
        
        # print("aaaaaaaaaaaaaaaaaaaaaaaa")

        s_gen = DataGen(data_path, self.buckets, epochs=1, max_width=self.max_original_width)
        for batch in tqdm(s_gen.gen(1), unit='batch'):
            current_step += 1
            # Get a batch (one image) and make a step.
            start_time = time.time()
            result = self.step(batch, self.forward_only)
            curr_step_time = (time.time() - start_time)

            num_total += 1

            output = result['prediction']
            ground = batch['labels'][0]
            # print("aaaaaaaaaaaaaaaaaaaaa")
            # print("ground" , ground)
            comment = batch['comments'][0]
            if sys.version_info >= (3,):
                output = output.decode('utf-8')
                ground = ground.decode('utf-8')
                comment = comment.decode('utf-8')

            probability = result['probability']

            if self.use_distance:
                incorrect = distance.levenshtein(output, ground)
                if len(ground) == 0:
                    if len(output) == 0:
                        incorrect = 0
                    else:
                        incorrect = 1
                else:
                    incorrect = float(incorrect) / len(ground)
                incorrect = min(1, incorrect)
            else:
                incorrect = 0 if output == ground else 1

            num_correct += 1. - incorrect

            if self.visualize:
                # Attention visualization.
                threshold = 0.5
                normalize = True
                binarize = True
                attns = np.array([[a.tolist() for a in step_attn] for step_attn in result['attentions']]).transpose([1, 0, 2])
                visualize_attention(batch['data'],
                                    'out',
                                    attns,
                                    output,
                                    self.max_width,
                                    DataGen.IMAGE_HEIGHT,
                                    threshold=threshold,
                                    normalize=normalize,
                                    binarize=binarize,
                                    ground=ground,
                                    flag=None)

            step_accuracy = "{:>4.0%}".format(1. - incorrect)
            correctness = step_accuracy + (" ({} vs {}) {}".format(output, ground, comment) if incorrect else " (" + ground + ")")

            tqdm.write('Step {:.0f} ({:.3f}s). Accuracy: {:6.2%}, loss: {:f}, perplexity: {:0<7.6}, probability: {:6.2%} {}'.format(
                current_step,
                curr_step_time,
                num_correct / num_total,
                result['loss'],
                math.exp(result['loss']) if result['loss'] < 300 else float('inf'),
                probability,
                correctness))

    def train(self, data_path, num_epoch, val_path):
        s_gen = DataGen(data_path, self.buckets, epochs=num_epoch, max_width=self.max_original_width)
        v_gen = DataGen(val_path, self.buckets, epochs=num_epoch, max_width=self.max_original_width)
        step_time = 0.0
        loss = 0.0
        current_step = 0
        num_total = 0.0
        num_correct = 0.0

        writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'train'),
                                       self.sess.graph)

        for batch in tqdm(s_gen.gen(self.batch_size), unit='batch'):
            current_step += 1

            start_time = time.time()
            result = self.step(batch, self.forward_only)
            loss += result['loss'] / self.steps_per_checkpoint
            curr_step_time = (time.time() - start_time)
            step_time += curr_step_time / self.steps_per_checkpoint

            writer.add_summary(result['summaries'], current_step)

            # precision = num_correct / len(batch['labels'])
            step_perplexity = math.exp(result['loss']) if result['loss'] < 300 else float('inf')

            tqdm.write('Step %i: %.3fs, loss: %f, perplexity: %f.'
                       % (current_step, curr_step_time, result['loss'], step_perplexity))

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % self.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                # Print statistics for the previous epoch.
                tqdm.write("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                           % (self.sess.run(self.global_step), step_time, loss, perplexity))
                # Save checkpoint and reset timer and loss.
                tqdm.write("Saving the model at step %d."%current_step)
                self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)
                step_time, loss = 0.0, 0.0

                if os.path.exists(val_path):
                    for val_batch in tqdm(v_gen.gen(self.batch_size), unit='batch'):
                        num_total += 1
                        val_result = self.step(val_batch, self.forward_only)
                        val_outputs = val_result['prediction']
                        val_grounds = val_batch['labels']
                        # print("aaaaaaaaaaaaaa")
                        # print(val_outputs)
                        # print("len(val_ground) = ", len(val_grounds))
                        # print("len(val_output) = ", len(val_outputs))
                        for i in range(len(val_outputs)): 
                            # print("index = ", i)                                 
                            if sys.version_info >= (3,):
                                val_output = val_outputs[i].decode('utf-8')
                                val_ground = val_grounds[i].decode('utf-8')
                                # print("bbbbbbbbbbbbb")
                                # print("val_output = ", val_output)
                                # print("val_ground = ", val_ground)
                                # print("val_output = {:30}, val ground = {}".format(val_output, val_ground))
                            if self.use_distance:
                                incorrects = 0
                                incorrect = distance.levenshtein(val_output, val_ground)
                                # print(incorrect)
                                if len(val_ground) == 0:
                                    if len(val_output) == 0:
                                        incorrect = 0
                                    else:
                                        incorrect = 1
                                else:
                                    incorrect = float(incorrect) / len(val_ground)
                                incorrects += incorrect
                                
                            else:
                                incorrect = 0 if val_output == val_ground else 1
                        incorrects /= 100
                        incorrects = min(1, incorrect)
                        num_correct += 1. - incorrect
                        tqdm.write("Accuracy: %f, loss: %f"
                            % (num_correct / num_total, result['loss']))
        # validation = []
        # validation.append(tf.summary.scalar("validation", num_correct / num_total))
        # all_summaries = tf.summary.merge(validation)
        # self.summaries_by_bucket.append(all_summaries)


        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        tqdm.write("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                   % (self.sess.run(self.global_step), step_time, loss, perplexity))

        self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)

    # step, read one batch, generate gradients
    def step(self, batch, forward_only):
        img_data = batch['data']
        decoder_inputs = batch['decoder_inputs']
        target_weights = batch['target_weights']

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl.name] = img_data

        for l in xrange(self.decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [
            self.attention_decoder_model.loss,  # Loss for this batch.
        ]

        # if not forward_only:
        #     output_feed += [self.summaries_by_bucket[0],
        #                     self.updates[0]]
        # else:
        #     output_feed += [self.prediction]
        #     output_feed += [self.probability]
        #     if self.visualize:
        #         output_feed += self.attention_decoder_model.attentions

        # outputs = self.sess.run(output_feed, input_feed)

        # res = {
        #     'loss': outputs[0],
        # }

        # if not forward_only:
        #     res['summaries'] = outputs[1]
        # else:
        #     res['prediction'] = outputs[1]
        #     res['probability'] = outputs[2]
        #     if self.visualize:
        #         res['attentions'] = outputs[3:]

        # if not forward_only:
        output_feed += [self.summaries_by_bucket[0],
                        self.updates[0]]
        # else:
        output_feed += [self.prediction]
        output_feed += [self.probability]
        if self.visualize:
            output_feed += self.attention_decoder_model.attentions

        outputs = self.sess.run(output_feed, input_feed)

        # # print(len(outputs))
        # print("\n\n\n\n\n\n\n")
        # # print(outputs[1])
        # # print("len output = %d" % len(outputs))
        # # print("len output_feed = %d" % len(output_feed))
        # # print("len input_feed = %d" % len(input_feed))

        # print("\n\n\n\n\n\n")

        # print(outputs[2])
        # print(outputs[3])
        # print(outputs[4])
        # print("\n\n\n\n\n\n\n")
        # print(input_feed)

        res = {
            'loss': outputs[0],
        }

        # if not forward_only:
        res['summaries'] = outputs[1]
        # else:
        res['prediction'] = outputs[3]
        res['probability'] = outputs[4]
        if self.visualize:
            res['attentions'] = outputs[5:]

        return res

    def _prepare_image(self, image):
        """Resize the image to a maximum height of `self.height` and maximum
        width of `self.width` while maintaining the aspect ratio. Pad the
        resized image to a fixed size of ``[self.height, self.width]``."""
        img = tf.image.decode_png(image, channels=self.channels)
        dims = tf.shape(img)
        self.width = self.max_width

        max_width = tf.to_int32(tf.ceil(tf.truediv(dims[1], dims[0]) * self.height_float))
        max_height = tf.to_int32(tf.ceil(tf.truediv(self.width, max_width) * self.height_float))

        resized = tf.cond(
            tf.greater_equal(self.width, max_width),
            lambda: tf.cond(
                tf.less_equal(dims[0], self.height),
                lambda: tf.to_float(img),
                lambda: tf.image.resize_images(img, [self.height, max_width],
                                               method=tf.image.ResizeMethod.BICUBIC),
            ),
            lambda: tf.image.resize_images(img, [max_height, self.width],
                                           method=tf.image.ResizeMethod.BICUBIC)
        )

        padded = tf.image.pad_to_bounding_box(resized, 0, 0, self.height, self.width)
        return padded
