import argparse

import tensorflow as tf

from model.model import Model
import config


def main(args):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = Model(phase='test',
                      visualize=args.visualize,
                      output_dir=args.output_dir,
                      batch_size=args.batch_size,
                      initial_learning_rate=args.initial_learning_rate,
                      steps_per_checkpoint=None,
                      model_dir=args.model_dir,
                      target_embedding_size=args.target_embedding_size,
                      attn_num_hidden=args.attn_num_hidden,
                      attn_num_layers=args.attn_num_layers,
                      clip_gradients=args.clip_gradients,
                      max_gradient_norm=args.max_gradient_norm,
                      session=sess,
                      load_model=True,
                      gpu_id=args.gpu_id,
                      use_gru=args.use_gru,
                      use_distance=args.use_distance,
                      max_image_width=args.max_width,
                      max_image_height=args.max_height,
                      max_prediction_length=args.max_prediction,
                      channels=args.channels)

        model.test(data_path=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max_width', type=int, default=config.MAX_WIDTH,
                        help='Max image width')
    parser.add_argument('--max_height', type=int, default=config.MAX_HEIGHT,
                        help='Max image height')
    parser.add_argument('--max_prediction', type=int, default=config.MAX_PREDICTION,
                        help='Max length of predicted strings')
    parser.add_argument('--channels', type=int, default=config.CHANNEL,
                        help='Color channel')
    parser.add_argument('--no_distance', dest="use_distance", action="store_false",
                        help='Require full match when calculating accuracy')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Specify a GPU ID')
    parser.add_argument('--use_gru', action='store_true',
                        help='use GRU instead of LSTM')
    parser.add_argument('--attn_num_layers', type=int, default=2,
                        help='Hidden layers in attention decoder cell')
    parser.add_argument('--attn_num_hidden', type=int, default=128,
                        help='Hidden units in attention decoder cell')
    parser.add_argument('--initial_learning_rate', type=float, default=1.0,
                        help='Initial learning rate')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help='Directory for model')
    parser.add_argument('--target_embedding_size', type=int, default=10,
                        help='Embedding dimension for each target')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--max_gradient_norm', type=int, default=5.0,
                        help='Clip gradients to this norm')
    parser.add_argument('--no_gradient_clipping', dest='clip_gradients', action='store_false',
                        help='Do not perform gradient clipping')

    parser.add_argument('--dataset', type=str,
                        help='Testing dataset in the TFRecords format')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    print(args)
    main(args)
