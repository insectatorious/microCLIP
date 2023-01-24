# Simple utils for loading CC3M data using tf.data for max performance.
#
# Each folder in the data directory should
# be treated as a SHARD. Each shard contains a number of files. A single
# training example consists of a .png image and a .txt file containing the
# corresponding label. The PNG and TXT files share the same name which
# should be used for matching the image to the corresponding caption.
# The label is a plain text string. The image is a 256x256x3 RGB image.

import os
import argparse
from functools import partial
from typing import Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


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


def load_dataset(data_dir, text_tokenizer, batch_size: Optional[int] = 32, shuffle_buffer_size=1000):
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
  dataset = load_dataset(data_dir, text_tokenizer, batch_size=None, shuffle_buffer_size=shuffle_buffer_size)

  # Create a new tfrecord writer
  writer = tf.io.TFRecordWriter(output_path,
                                options=tf.io.TFRecordOptions(compression_type='GZIP'))

  # Count the total number of items in the dataset
  count = tf.data.experimental.cardinality(dataset).numpy()

  # Use tqdm to display a progress bar
  with tqdm(total=count) as pbar:
    # Iterate through the dataset and write the data to the tfrecord file
    for (label, image) in dataset:
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

  # Close the tfrecord writer
  writer.close()


def load_cc3m(data_dir, text_tokenizer, batch_size=32, shuffle_buffer_size=1000):
  """Load the CC3M dataset."""
  return load_dataset(data_dir, text_tokenizer, batch_size, shuffle_buffer_size)


def read_tfrecord(file_path, batch_size=32, image_size=(256, 256)):
  # Create a dataset from the tfrecord file
  dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')

  # Define a function to parse the tfrecord example
  def parse_example(example_proto):
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
    # example['image'] = tf.reshape(example['image'], (image_size[0], image_size[1], 3))
    # example['image'] = tf.image.convert_image_dtype(example['image'], tf.float32)

    return example['label'], example['image']

  # Map the parse function to the dataset
  dataset = dataset.map(parse_example)
  # Batch the dataset
  dataset = dataset.batch(batch_size, drop_remainder=True)
  # Prefetch the dataset
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return dataset


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

  # Write the dataset to a tfrecord file
  load_cc3m_to_tfrecord(args.data_dir,
                        text_tokenizer,
                        batch_size=args.batch_size,
                        shuffle_buffer_size=args.shuffle_buffer_size,
                        output_path=args.output_path)
