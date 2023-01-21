import os

import tensorflow as tf


class ImageTextCosineSimilarityCallback(tf.keras.callbacks.Callback):
  """Callback for computing cosine similarity between image and text embeddings."""

  def __init__(self, images, texts_tokenised, tensorboard_log_dir):
    super().__init__()
    self.images = images
    self.texts_tokenised = texts_tokenised
    self.tensorboard_log_dir = os.path.join(tensorboard_log_dir, "cosine_similarity")
    self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_log_dir)

  def on_epoch_end(self, epoch, logs=None):
    logits = self.model((self.texts_tokenised, self.images))
    logits_softmax = tf.nn.softmax(logits / self.model.temperature, axis=-1)

    # print(f'Cosine similarity: {cosine_similarity.numpy()}')
    with self.tensorboard_writer.as_default():
      # Log cosine similarity separately for each image
      for i, label in zip(range(logits.shape[0]), ["bus", "cat", "dog"]):
        tf.summary.scalar(f"cosine_similarity/softmax/{label}", logits_softmax[i, i], step=epoch)
        tf.summary.scalar(f"cosine_similarity/raw/{label}", logits[i, i], step=epoch)