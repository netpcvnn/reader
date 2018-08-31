# Word Reader

## Installation

+ Requirements:
  * Python >= 3.4
  * Tensorflow >= 1.4.0
  * tqdm

+ Config
  * Copy `.env.conf` to `.env`, make necessary config and `source .env`


## Data

+ Prepare annotation file, i.e text file containing the image paths
(relative path from top level of this project) and their corresponding labels:

```
dataset/images/hello.jpg hello
dataset/images/world.jpg world
```

+ (Optional) To prepare notation file, just name file as follow syntax:

```
label_imgname.jpg
```

and then use this script:

```
python data/prepare_notation.py --dataset_dir dataset/train/ --out dataset/train.txt
```

+ Build a TFRecords dataset. You need a collection of images and annotation files as mentioned above:

```
python src/dataset.py --annotation ./dataset/annotations-training.txt --output ./dataset/training.tfrecords
python src/dataset.py --annotation ./dataset/annotations-testing.txt --output ./dataset/testing.tfrecords
```

## Train

```
python src/train.py --dataset ./dataset/training.tfrecords --batch_size 100
```

Supervise with tensorboard: `tensorboard --logdir checkpoints/train` (check http://localhost:6006)


## Test

```
python src/test.py --dataset ./datasets/testing.tfrecords
```

Additionally, you can visualize the attention results during testing (saved to `results/` by default):

```
python src/test.py --dataset ./datasets/testing.tfrecords --visualize
```


## Predict

### Single predict

```
python src/predict.py
```

Input image paths line by line for corresponding prediction with probability.

### Batch predict

+ Prepare file containing all image paths (refer annotation without label):

```
dataset/images/hello.jpg
dataset/images/world.jpg
```

+ Create TFRecords dataset:

```
python src/dataset.py --predict --annotation ./datasets/predict.txt --output ./dataset/predicting.tfrecords
```

+ Script:

```
python src/batch_predict.py --dataset ./dataset/predicting.tfrecrods --batch_size 100
```


## Export

Exporte graph for Tensorflow Serving

```
python src/export.py --export_dir /path/to/export/dir --model_dir /path/to/checkpoint  --version $VERSION
```

*Example*: Export name reader checkpoint, version 3

```
python src/export.py --export_dir /home/ducna/share/Reader/name --model_dir checkpoints/name --version 3
```


## References

+ Word Reader is heavily based from https://github.com/emedvedev/attention-ocr
