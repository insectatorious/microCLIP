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


class BatchMetricsCallback(tf.keras.callbacks.Callback):
  """Callback for writing loss scalar metrics every N batches to Tensorboard."""

  def __init__(self, tensorboard_log_dir, batch_interval=25):
    super().__init__()
    self.tensorboard_log_dir = tensorboard_log_dir
    self.batch_interval = batch_interval
    self.tensorboard_loss_writer = tf.summary.create_file_writer(
      os.path.join(tensorboard_log_dir, "loss")
    )
    self.tensorboard_logits_writer = tf.summary.create_file_writer(
      os.path.join(tensorboard_log_dir, "logits")
    )
    # on
    self.epoch = -1

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch

  def on_train_batch_end(self, batch, logs=None):
    batch += self.params["steps"] * self.epoch
    if batch % self.batch_interval == 0:
      with self.tensorboard_loss_writer.as_default():
        self.write_loss_metrics(batch, logs)
      with self.tensorboard_logits_writer.as_default():
        self.write_logits_metrics(batch, logs)

  def on_test_batch_end(self, batch, logs=None):
    if batch % self.batch_interval == 0:
      with self.tensorboard_loss_writer.as_default():
        self.write_loss_metrics(batch, logs)
      with self.tensorboard_logits_writer.as_default():
        self.write_logits_metrics(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    if batch % self.batch_interval == 0:
      with self.tensorboard_logits_writer.as_default():
        self.write_logits_metrics(batch, logs)

  @staticmethod
  def write_loss_metrics(batch, logs):
    tf.summary.scalar("loss/total", logs["loss"], step=batch)
    tf.summary.scalar("loss/image", logs["image_loss"], step=batch)
    tf.summary.scalar("loss/text", logs["text_loss"], step=batch)

  @staticmethod
  def write_logits_metrics(batch, logs):
    tf.summary.histogram("logits/total", logs["logits"], step=batch)
    tf.summary.scalar("logits/max", tf.reduce_max(logs["logits"]), step=batch)
    tf.summary.scalar("logits/mean", tf.reduce_mean(logs["logits"]), step=batch)
    tf.summary.scalar("logits/min", tf.reduce_min(logs["logits"]), step=batch)
