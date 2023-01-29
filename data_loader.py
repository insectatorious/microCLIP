# Simple utils for loading CC3M data using tf.data for max performance.
#
# Each folder in the data directory should
# be treated as a SHARD. Each shard contains a number of files. A single
# training example consists of a .png image and a .txt file containing the
# corresponding label. The PNG and TXT files share the same name which
# should be used for matching the image to the corresponding caption.
# The label is a plain text string. The image is a 256x256x3 RGB image.

import os
import random
import argparse
from functools import partial
from typing import Optional, List, Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

MEAN_RGB = [0.6525933, 0.6365939, 0.61723816]


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2) -> tf.Tensor:
  gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1, dtype=tf.float32)
  gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0, dtype=tf.float32)
  return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def load_image(image_path, label_path, encode_fn) -> Tuple[Optional[tf.Tensor],
                                                           Optional[tf.Tensor]]:
  """Load an image and its corresponding label."""
  try:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
  except tf.errors.InvalidArgumentError as ex:
    print("Failed to load image: {}".format(image_path))
    return None, None

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


def load_dataset(data_dir,
                 text_tokenizer,
                 batch_size: Optional[int] = 32,
                 shuffle_buffer_size=1000):
  """Load a dataset from a directory."""
  # Curry the load_image function to pass the text_tokenizer.
  encode_fn = partial(encode_text, tokeniser=text_tokenizer)
  load_image_fn = partial(load_image, encode_fn=encode_fn)
  image_paths = []
  label_paths = []
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      if file.endswith(".png"):
        expected_label_path = os.path.join(root, file.replace(".png", ".txt"))
        if not os.path.exists(expected_label_path):
          print("Missing label file: {}".format(expected_label_path))
          continue
        label_paths.append(expected_label_path)
        image_paths.append(os.path.join(root, file))

  image_paths = tf.constant(image_paths)
  label_paths = tf.constant(label_paths)
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.map(load_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.filter(lambda label, image: label is not None)
  if batch_size:
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return dataset


def load_cc3m_to_tfrecord(data_dir,
                          text_tokenizer,
                          batch_size=32,
                          shuffle_buffer_size=1000,
                          output_path='cc3m.tfrecord'):
  """Load the CC3M dataset and write it to a tfrecord file."""
  dataset = load_dataset(data_dir,
                         text_tokenizer,
                         batch_size=batch_size,
                         shuffle_buffer_size=shuffle_buffer_size)

  # Shard the dataset into multiple files
  dataset = dataset.shard(10, 0)

  # Create a new tfrecord writer
  writer = tf.io.TFRecordWriter(output_path,
                                options=tf.io.TFRecordOptions(compression_type='GZIP',
                                                              compression_level=9))

  # Count the total number of items in the dataset
  count = tf.data.experimental.cardinality(dataset).numpy()

  # Use tqdm to display a progress bar
  with tqdm(total=count) as pbar:
    # Iterate through the dataset and write the data to the tfrecord file,
    # handling cases where the PNG is corrupt.
    for label, image in dataset:
      try:
        # Create a feature dictionary for the label and image data
        feature = {
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.numpy().flatten())),
          'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.numpy().flatten()))
          # 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()]))
        }
        # Create an example message from the feature dictionary
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Write the example message to the tfrecord file
        writer.write(example.SerializeToString())
        # Update the progress bar
        pbar.update(1)
      except tf.errors.InvalidArgumentError as ex:
        print(f"Failed to write example: {ex}")
        continue

  # Close the tfrecord writer
  writer.close()


def load_cc3m(data_dir, text_tokenizer, batch_size=32, shuffle_buffer_size=1000):
  """Load the CC3M dataset."""
  return load_dataset(data_dir, text_tokenizer, batch_size, shuffle_buffer_size)


def read_tfrecord(file_path, batch_size: Optional[int] = 32, image_size=(256, 256), subtract_mean=True):
  # Create a dataset from the tfrecord file
  dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')

  # Define a function to parse the tfrecord example
  def parse_example(example_proto, image_size=image_size):
    example = parse_label_image_proto(example_proto, image_size=image_size, subtract_mean=subtract_mean)

    return example['label'], example['image']

  # Map the parse function to the dataset
  dataset = dataset.map(parse_example)
  if batch_size:
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return dataset


def parse_label_image_proto(example_proto, image_size=(256, 256), subtract_mean=True):
  # Define the features to parse from the tfrecord
  feature_description = {
    'label': tf.io.FixedLenFeature(shape=(128,), dtype=tf.int64),
    'image': tf.io.FixedLenFeature([256 * 256 * 3], dtype=tf.float32),
  }
  # Parse the example using the feature description
  example = tf.io.parse_single_example(example_proto, feature_description)
  # decode the image which is stored as a list of floats
  example['image'] = tf.reshape(example['image'], (256, 256, 3))
  # reshape the image to the desired size
  example['image'] = tf.image.resize(example['image'], image_size)

  # subtract the mean from the image
  if subtract_mean:
    example['image'] -= tf.constant(MEAN_RGB, dtype=tf.float32)
  # example['image'] = tf.reshape(example['image'], (image_size[0], image_size[1], 3))
  # example['image'] = tf.image.convert_image_dtype(example['image'], tf.float32)
  return example


def mix_up_datasets(ds_one, ds_two, alpha: float = 0.2):
  """Mix two CC3M datasets together.
  It is assumed that
  - the datasets have the same number of elements
  - the datasets have not been batched
  - the datasets have the same batch size
  """
  # Unpack the datasets
  ds_one_labels, ds_one_images = ds_one
  ds_two_labels, ds_two_images = ds_two
  batch_size = tf.shape(ds_one_labels)[0]

  # Sample lambda and reshape it to do the mixup
  l = sample_beta_distribution(batch_size, alpha, alpha)
  x_l = tf.reshape(l, (batch_size, 1, 1, 1))
  y_l = tf.reshape(l, (batch_size, 1))

  # Cast labels to float32
  ds_one_labels = tf.cast(ds_one_labels, tf.float32)
  ds_two_labels = tf.cast(ds_two_labels, tf.float32)

  # Mix the datasets together
  images = ds_one_images * x_l + ds_two_images * (1 - x_l)
  labels = ds_one_labels * y_l + ds_two_labels * (1 - y_l)

  return (labels, images), l


if __name__ == '__main__':
  # Get arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data/cc3m')
  parser.add_argument('--output_path', type=str, default='data/cc3m.tfrecord')
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
  args = parser.parse_args()

  # Test the data loader.
  from text_transformer.model import TextTransformer

  text_transformer = TextTransformer(64, num_heads=4, num_layers=4, ff_dim=128)
  text_tokenizer = text_transformer.tokenizer

  # Write the dataset to a tfrecord file, catching any errors
  try:
    load_cc3m_to_tfrecord(args.data_dir,
                          text_tokenizer,
                          batch_size=args.batch_size,
                          shuffle_buffer_size=args.shuffle_buffer_size,
                          output_path=args.output_path)
  except Exception as e:
    print(e)
