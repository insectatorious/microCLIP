import json
import os
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from clearml import Task

from callbacks import ImageTextCosineSimilarityCallback, BatchMetricsCallback
from data_loader import read_tfrecord, mix_up_datasets
from model import MicroCLIP
from image_encoder.model import ResNet
from text_transformer.model import TextTransformer


# DATA_DIR = os.environ.get("DATA_DIR", '/Users/sumanas/Documents/data/cc3m')


def main(config):
  text_transformer = TextTransformer(embedding_dim=config['text_embedding_dim'],
                                     num_heads=config['text_num_heads'],
                                     num_layers=config['text_num_layers'],
                                     ff_dim=config['text_ff_dim'],
                                     num_classes=None)
  image_encoder = ResNet(filters=config['img_filter_count'],
                         num_classes=None)
  image_encoder.build((None, 64, 64, 3))

  # Fetch test images from Imgur
  urls = [
    "https://i.imgur.com/y8riKYY.jpg",  # bus
    "https://i.imgur.com/Tm45pBV.jpg",  # cat
    "https://i.imgur.com/86KDlmh.jpg",  # dog
  ]
  images = [tf.keras.utils.get_file(origin=url) for url in urls]
  images = [tf.keras.preprocessing.image.load_img(image, target_size=(64, 64)) for image in images]
  images = [tf.keras.preprocessing.image.img_to_array(image) for image in images]
  images = np.asarray(images) / 255.0
  texts = ["a photo of a bus", "a photo of a cat", "a photo of a dog"]
  texts_tokenised = np.asarray([text_transformer.tokenizer.encode(text) for text in texts])

  logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

  clip = MicroCLIP(image_encoder=image_encoder,
                   text_encoder=text_transformer,
                   # temperature=config["temperature"],
                   latent_dim=config["latent_dim"],
                   mixup=config["mixup"], )

  dataset = read_tfrecord(config["tfrecord_path"],
                          batch_size=config["batch_size"],
                          image_size=(64, 64))
  if config["mixup"]:
    dataset = tf.data.Dataset.zip((dataset, dataset.shuffle(100)))
    dataset = dataset.map(lambda x, y: mix_up_datasets(x, y, alpha=0.2),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

  # dataset = dataset.take(1000)
  clip.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=None,
  )

  callbacks = [
    ImageTextCosineSimilarityCallback(texts_tokenised, images, logdir),
    BatchMetricsCallback(logdir),
  ]
  if config["reduce_lr"]:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                          factor=0.2,
                                                          patience=1,
                                                          min_lr=1e-6))
  clip.fit(dataset,
           epochs=config["epochs"],
           callbacks=callbacks, )

  clip.save('clip')
  clip.save_weights('clip_weights.h5')

  return clip


if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  # parser.add_argument('--data_dir', required=True, type=str, help='Path to CC3M dataset')
  parser.add_argument('--tfrecord_path', required=True, type=str, help='Path to TFRecord dataset')
  parser.add_argument('--epochs', required=False, type=int, default=100, help='Number of epochs to train for')
  parser.add_argument('--batch_size', required=False, type=int, default=128, help='Batch size')
  parser.add_argument('--temperature', required=False, type=float, default=0.1, help='Temperature for softmax')
  parser.add_argument('--img_filter_count', required=False, type=int, default=128,
                      help='Number of filters in image encoder')
  parser.add_argument('--text_embedding_dim', required=False, type=int, default=128,
                      help='Embedding dimension for text encoder')
  parser.add_argument('--latent_dim', required=False, type=int, default=128,
                      help='Latent dimension for both encoders')
  parser.add_argument('--text_num_heads', required=False, type=int, default=4,
                      help='Number of attention heads for text encoder')
  parser.add_argument('--text_num_layers', required=False, type=int, default=4,
                      help='Number of layers for text encoder')
  parser.add_argument('--text_ff_dim', required=False, type=int, default=512,
                      help='Feed forward dimension for text encoder')
  parser.add_argument('--reduce_lr', required=False, action='store_true',
                      help='Reduce LR on plateau')
  parser.add_argument('--mixup', required=False, action='store_true',
                      help='Use Mixup technique for training')
  parser.add_argument('--weights_path', required=False, type=str, default='clip_weights.h5',
                      help='Path to weights file to save to')

  args = parser.parse_args()
  config = vars(args)
  task = Task.init(project_name='microCLIP', task_name='local training')
  task.connect({k: v for k, v in config.items()
                if k in ['tfrecord_path', 'epochs', 'batch_size',
                         'img_filter_count', 'text_embedding_dim', 'latent_dim',
                         'text_num_heads', 'text_num_layers', 'text_ff_dim',
                         'reduce_lr', 'mixup', 'weights_path']})
  # Save config to file
  with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
  clip = main(config)
