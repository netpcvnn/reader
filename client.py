import argparse
import sys
import os

import tensorflow as tf
from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def main(args):
    # Connect to server
    host, port = args.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    for img_name in os.listdir(args.dataset_dir):
        # Prepare request object
        request = predict_pb2.PredictRequest()
        request.model_spec.name = args.model_name
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        # Get data
        img_path = os.path.join(args.dataset_dir, img_name)
        try:
            with open(img_path, 'rb') as f:
                img = f.read()
        except IOError:
            print('[ERROR] Error opening {}'.format(img_path))
            continue

        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(img))

        # Do inference
        result_future = stub.Predict.future(request, 5.0)

        # Extract output
        word = result_future.result().outputs['output'].string_val[0]
        if sys.version_info >= (3,):
            word = word.decode('utf-8')

        prob = result_future.result().outputs['probability'].double_val[0]

        print('{}\t{}\t{}'.format(img_name, word, prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='localhost:9000')
    parser.add_argument('--model_name', type=str, default='reader')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
