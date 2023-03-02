import os
from typing import Optional, List

import tensorflow as tf

from image_encoder.model import ResNet
from text_transformer.model import TextTransformer


class MicroCLIP(tf.keras.Model):
  def __init__(self,
               text_encoder: Optional[tf.keras.Model] = None,
               image_encoder: Optional[tf.keras.Model] = None,
               latent_dim: int = 512,
               # temperature: float = 1.0,
               mixup: bool = False,
               **kwargs):
    super().__init__(**kwargs)

    # Handle case where text_encoder and image_encoder are None
    if text_encoder is None:
      text_encoder = TextTransformer(64, num_heads=4, num_layers=4, ff_dim=128, num_classes=None)
    if image_encoder is None:
      image_encoder = ResNet(filters=64, num_classes=None)

    assert isinstance(text_encoder, tf.keras.Model)
    assert isinstance(image_encoder, tf.keras.Model)
    assert hasattr(text_encoder, "tokenizer")
    assert hasattr(image_encoder, "image_preprocessor")

    self.text_encoder = text_encoder
    self.image_encoder = image_encoder
    self.image_preprocessor = image_encoder.image_preprocessor
    self.temperature = tf.Variable(0.1, trainable=True, name="temperature")
    self.latent_dim = latent_dim
    self.mixup = mixup

    self.text_linear_projection = tf.keras.layers.Dense(latent_dim, use_bias=False)
    self.image_linear_projection = tf.keras.layers.Dense(latent_dim, use_bias=False)

  def call(self, inputs, training=False, mask=None):
    """Performs a forward pass.

    Computes the text and image encodings and returns the logits for the
    cross-modal similarity using dot-product attention.
    Cosine similarity is used to compute the logits. The logits are
    normalised to be in the range [0, 1].

    The text input can be one of the following:
    - a list of strings
    - a dict with keys `input_ids` and `attention_mask`
    - a tuple with two elements: `input_ids` and `attention_mask`

    Args:
      inputs: A tuple of (text, image) where text is a string and image is a
        tf.Tensor of shape (batch_size, image_edge, image_edge, 3).
      training: Whether the model is in training mode.
      mask: A mask for the inputs.

    Returns:
      A tf.Tensor of shape (batch_size, batch_size) containing the logits for
      the cross-modal similarity. The logits are normalised to be in the range
      [0, 1]. The logits are computed using cosine similarity. SOFTMAX is not
      applied to the logits. Shape of logits: (batch_size, batch_size)
    """
    text, image = inputs
    text_features = self.text_encoder(text, training=training)
    text_features = self.text_linear_projection(text_features)
    image_features = self.image_encoder(image, training=training)
    image_features = self.image_linear_projection(image_features)

    # normalized features
    image_features = tf.math.l2_normalize(image_features, axis=-1)
    text_features = tf.math.l2_normalize(text_features, axis=-1)

    # cosine similarity as logits
    logit_scale = tf.math.exp(self.temperature)
    logits_per_image = logit_scale * image_features @ tf.transpose(text_features)
    logits_per_text = logit_scale * text_features @ tf.transpose(image_features)
    # logits_per_text = logit_scale * tf.matmul(text_features, tf.transpose(image_features))

    # shape = [global_batch_size, global_batch_size]
    return logits_per_text, logits_per_image

    # image_features /= tf.linalg.normalize(image_features, axis=-1)
    # text_features /= tf.linalg.normalize(text_features, axis=-1)
    #
    # # Joint multi-modal embedding
    # # image_logits = tf.tensordot(image_features, text_features, axes=1)
    # image_logits = tf.einsum("nc,nc->n", image_features, text_features)
    # normalized_image_logits = tf.linalg.normalize(image_logits, axis=0)
    # # text_logits = tf.tensordot(text_features, image_features, axes=1)
    # text_logits = tf.einsum("nc,nc->n", text_features, image_features)
    # normalized_text_logits = tf.linalg.normalize(text_logits, axis=0)
    #
    # logits = tf.tensordot(normalized_image_logits, normalized_text_logits, axes=0)
    # logits *= tf.exp(self.temperature)
    #
    # return logits

  def get_config(self):
    return {"text_encoder": self.text_encoder,
            "image_encoder": self.image_encoder,
            "latent_dim": self.latent_dim,
            # "temperature": self.temperature,
            "mixup": self.mixup}

  def train_step(self, data):
    """Performs a training step.

    Use cross-entropy loss to train the model. The logits are not normalised
    using softmax. The logits are normalised using cosine similarity.

    Args:
      data: A tuple of (text, image) where text is a string and image is a
        tf.Tensor of shape (batch_size, image_edge, image_edge, 3).

    Returns:
      A dictionary containing the loss and other metrics.
    """
    # text, image = data
    if self.mixup:
      text, image = data[0]
      lambdas = data[1]
    else:
      text, image = data
    with tf.GradientTape() as tape:

      logits_per_text, logits_per_image = self((text, image), training=True)
      # logits = self((text, image), training=True)
      labels = tf.eye(tf.shape(logits_per_text)[0], dtype=tf.float32)
      if self.mixup:
        labels *= lambdas
      image_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_per_image)
      text_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_per_text)
        # image_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True, axis=0)
      # text_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True, axis=1)
      loss = tf.reduce_mean(image_loss + text_loss)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return {"loss": loss,
            "image_loss": tf.reduce_mean(image_loss),
            "text_loss": tf.reduce_mean(text_loss),
            "logits_per_text": logits_per_text,
            "logits_per_image": logits_per_image}


if __name__ == '__main__':
  import numpy as np

  image = np.random.random((3, 256, 256, 3))
  model = MicroCLIP()
  print(model((["This is a cat"] * 3, image)))


def test_helper(image_path, texts: List[str], clip_model: tf.keras.Model):
  from PIL import Image

  # np.random.shuffle(texts)
  image = Image.open(image_path)
  image = image.resize((64, 64))
  image = np.asarray(image).astype(np.float32) / 255.0
  texts_tokenised = np.asarray([clip_model.tokenizer.encode(text) for text in texts])
  logits = clip_model((texts_tokenised, np.repeat([image], 5, axis=0)))
  print(logits)
  print(logits.shape)
  y_pred = tf.argmax(tf.nn.softmax(logits, axis=1))
  print(y_pred)
  print(texts[y_pred])
