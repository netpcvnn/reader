import argparse
import logging

import tensorflow as tf

from util import dataset


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')


def main(args):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        dataset.generate(args.annotation, args.output,
                         args.log_step, args.force_uppercase,
                         args.predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str,
                        help='Path to the annotation file')
    parser.add_argument('--output', type=str,
                        help='Output path for TFrecords')
    parser.add_argument('--predict', action='store_true',
                        help='Make dataset for batch predicting')
    parser.add_argument('--log_step', type=int, default=500,
                        help='print log messages every N steps')
    parser.add_argument('--no_force_uppercase', dest='force_uppercase', action='store_false',
                        help='Do not force uppercase on label values')

    args = parser.parse_args()
    main(args)
