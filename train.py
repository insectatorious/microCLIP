import os
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from callbacks import ImageTextCosineSimilarityCallback, BatchMetricsCallback
from data_loader import load_cc3m
from model import MicroCLIP
from image_encoder.model import ResNet
from text_transformer.model import TextTransformer

# DATA_DIR = os.environ.get("DATA_DIR", '/Users/sumanas/Documents/data/cc3m')


def main(config):
  text_transformer = TextTransformer(64, num_heads=4, num_layers=4, ff_dim=128, num_classes=None)
  image_encoder = ResNet(filters=64, num_classes=None)
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
                   temperature=config["temperature"])
  dataset = load_cc3m(config["data_dir"],
                      text_transformer.tokenizer,
                      batch_size=config["batch_size"],)

  clip.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=None,
  )

  clip.fit(dataset,
           epochs=config["epochs"],
           callbacks=[ImageTextCosineSimilarityCallback(images, texts_tokenised, logdir),
                      BatchMetricsCallback(logdir)])
  clip.save_weights('clip.h5')

  return clip


if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=str, help='Path to CC3M dataset')
  parser.add_argument('--epochs', required=False, type=int, default=100, help='Number of epochs to train for')
  parser.add_argument('--batch_size', required=False, type=int, default=128, help='Batch size')
  parser.add_argument('--temperature', required=False, type=float, default=0.1, help='Temperature for softmax')

  args = parser.parse_args()
  config = vars(args)
  clip = main(config)
