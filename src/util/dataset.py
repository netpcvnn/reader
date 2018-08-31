import tensorflow as tf
import logging
import cv2
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def preprocessing(img_path):
    image = cv2.imread(img_path, 0)

    print(image.shape)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    pro0 = clahe.apply(image)

    gaussian_3 = cv2.GaussianBlur(pro0, (9,9), 10.0)

    unsharp_image = cv2.addWeighted(pro0, 1.5, gaussian_3, -0.5, 0, pro0)

    ret, pro1 = cv2.threshold(pro0, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    _, blackAndWhite = cv2.threshold(pro1, 127, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, 2, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 15:   #filter small dotted regions
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)

    cv2.imshow("image", res)
    cv2.waitKey(1000)

    # cv2.imwrite(img_path, res)


def generate(annotations_path, output_path,
             log_step=5000, force_uppercase=True, predict=False):
    logging.info('Building a dataset from {}'.format(annotations_path))
    logging.info('Output file: {}'.format(output_path))


    print(annotations_path)
    i = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    with open(annotations_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n')
            # i+= 1
            # print(line)
            

            if not predict:
                # Train/Test
                try:
                    (img_path, label) = line.split('\t', 1)
                    img_path = '/home/hoangtienduc/vision/dataset/retrain_reader_17_8/name/accur/to_6_val/' + img_path
                    
                
                    # try:
                    #     preprocessing(img_path)
                    # except:
                    #     continue

                except ValueError:
                    logging.error('Missing filename or label, ignoring line {}: {}'.format(idx+1, line))
                    continue
                try:
                    with open(img_path, 'rb') as img_file:
                        img = img_file.read()
                    i += 1
                    print(img_path)
                    print("aaaaaaaaaaaaaaaaaa")
                    if force_uppercase:
                        label = label.upper()

                    feature = {
                        'image': _bytes_feature(img),
                        'label': _bytes_feature(label.encode('utf-8')),
                        'path': _bytes_feature(img_path.encode('utf-8'))
                    }
                except: 
                    continue
            else:
                # Batch predict
                img_path = line
                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                feature = {
                    'image': _bytes_feature(img),
                    'label': _bytes_feature(''.encode('utf-8')),
                    'path': _bytes_feature(img_path.encode('utf-8'))
                }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            if (idx+1) % log_step == 0:
                logging.info('Processed {} pairs'.format(idx+1))
    print("so luong anh : ", i)
    logging.info('Dataset is ready: {} pairs'.format(idx+1))
    writer.close()
