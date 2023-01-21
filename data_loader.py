# Simple utils for loading CC3M data using tf.data for max performance.
#
# Each folder in the data directory should
# be treated as a SHARD. Each shard contains a number of files. A single
# training example consists of a .png image and a .txt file containing the
# corresponding label. The PNG and TXT files share the same name which
# should be used for matching the image to the corresponding caption.
# The label is a plain text string. The image is a 256x256x3 RGB image.

import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.data.experimental import AUTOTUNE


def load_image(image_path, label_path, encode_fn):
  """Load an image and its corresponding label."""
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, (256, 256))
  label = tf.io.read_file(label_path)
  label = tf.strings.strip(label)

  # Tokenize the label.
  label = tf.py_function(encode_fn, inp=[label], Tout=tf.int32)
  label.set_shape([None])

  return label, image


def encode_text(text, tokeniser):
  # pad the text to a max length of 128
  if isinstance(text, tf.Tensor):
    text = text.numpy().decode("utf-8")
  encoded_text = tokeniser.encode(text)
  encoded_text = pad_sequences([encoded_text], maxlen=128, padding="post", truncating="post")
  return encoded_text[0]


def load_dataset(data_dir, text_tokenizer, batch_size=32, shuffle_buffer_size=1000):
  """Load a dataset from a directory."""
  # Curry the load_image function to pass the text_tokenizer.
  encode_fn = partial(encode_text, tokeniser=text_tokenizer)
  load_image_fn = partial(load_image, encode_fn=encode_fn)
  image_paths = []
  label_paths = []
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      if file.endswith(".png"):
        image_paths.append(os.path.join(root, file))
        expected_label_path = os.path.join(root, file.replace(".png", ".txt"))
        if not os.path.exists(expected_label_path):
          raise ValueError("Missing label file: {}".format(expected_label_path))
        label_paths.append(expected_label_path)

  image_paths = tf.constant(image_paths)
  label_paths = tf.constant(label_paths)
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.map(load_image_fn, num_parallel_calls=AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)

  return dataset


def load_cc3m(data_dir, text_tokenizer, batch_size=32, shuffle_buffer_size=1000):
  """Load the CC3M dataset."""
  return load_dataset(data_dir, text_tokenizer, batch_size, shuffle_buffer_size)


if __name__ == '__main__':
  # Test the data loader.
  from text_transformer.model import TextTransformer

  text_transformer = TextTransformer(64, num_heads=4, num_layers=4, ff_dim=128)
  dataset = load_cc3m('/Users/sumanas/Documents/data/cc3m', text_transformer.tokenizer)
  for label, image in dataset:
    print(image.shape)
    print(label.shape)
