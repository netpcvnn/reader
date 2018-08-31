import numpy as np
import tensorflow as tf
import sys

from .bucketdata import BucketData
from PIL import Image
from six import BytesIO as IO

# Call config from parent
sys.path.append('../')
import config


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32
    CHARMAP = ['', '', ''] + list(config.DICTIONARY)

    @staticmethod
    def setFullAsciiCharmap():
        DataGen.CHARMAP = ['', '', ''] + [chr(i) for i in range(32, 127)]

    def __init__(self,
                 annotation_fn,
                 buckets,
                 epochs=1000,
                 max_width=None):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.epochs = epochs
        self.max_width = max_width

        self.bucket_specs = buckets
        self.bucket_data = BucketData()

        dataset = tf.data.TFRecordDataset([annotation_fn])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = BucketData()

    def gen(self, batch_size):
        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        images, labels, comments, paths = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                try:
                    raw_images, raw_labels, raw_comments, raw_paths = sess.run([images, labels, comments, paths])
                    for img, lex, comment, path in zip(raw_images, raw_labels, raw_comments, raw_paths):
                        try:
                            #width_image = Image.open(IO(img)).size[0]. 

                            if self.max_width and (Image.open(IO(img)).size[0] <= self.max_width):
                                # print("########################")
                                # print(Image.open(IO(img)).size[0])
                                # print("########################")
                                word = self.convert_lex(lex)

                                bucket_size = self.bucket_data.append(img, word, lex, comment, path)
                                if bucket_size >= batch_size:
                                    bucket = self.bucket_data.flush_out(
                                        self.bucket_specs,
                                        go_shift=1)
                                    yield bucket
                        except:
                            continue
                except tf.errors.OutOfRangeError:
                    break

        self.clear()

    def convert_lex(self, lex):
        if sys.version_info >= (3,):
            lex = lex.decode('utf-8')

        assert len(lex) < self.bucket_specs[-1][1]

        return np.array(
            [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
            dtype=np.int32)

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'comment': tf.FixedLenFeature([], tf.string, default_value=''),
                'path': tf.FixedLenFeature([], tf.string)
            })
        return features['image'], features['label'], features['comment'], features['path']
