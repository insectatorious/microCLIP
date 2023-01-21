# Simple Keras implementation of a text transformer (Vaswani et al., 2017)
import time
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from simple_tokenizer import SimpleTokenizer


class TextTransformer(tf.keras.Model):
  def __init__(self,
               embedding_dim,
               num_heads,
               ff_dim,
               num_layers=4,
               rate=0.1,
               num_classes: Optional[int] = 1):
    super(TextTransformer, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.embedding_dim = embedding_dim
    self.tokenizer = SimpleTokenizer()
    self.vocab_size = len(self.tokenizer.encoder)
    self.embedding = layers.Embedding(self.vocab_size, embedding_dim)
    self.pos_encoding = positional_encoding(self.vocab_size, embedding_dim)
    self.enc_layers = [EncoderLayer(embedding_dim, num_heads, ff_dim, rate)
                       for _ in range(num_layers)]
    self.dropout = layers.Dropout(rate)
    self.pooling = layers.GlobalAveragePooling1D()
    if num_classes:
      self.linear_layer = layers.Dense(32, activation="gelu")
      self.output_layer = layers.Dense(num_classes, activation="sigmoid")

  def call(self, inputs, training=None, mask=None):
    # `inputs` can be one of the following:
    # - a string
    # - a list of strings
    # - tokenised input (i.e. a list of lists of integers)
    # - tokenised input in the form of a numpy array
    # - tokenised input in the form of a tf.Tensor
    # TODO: add support for other input types
    # if isinstance(inputs, str):
    #   inputs = [inputs]
    # if isinstance(inputs[0], str):
    #   inputs = tf.convert_to_tensor([self.tokenizer.encode(value) for value in inputs])
    # if isinstance(inputs, np.ndarray):
    #   inputs = tf.convert_to_tensor(inputs)
    # if isinstance(inputs, tf.Tensor):
    #   inputs = tf.convert_to_tensor(inputs)

    x = inputs
    seq_len = tf.shape(x)[1]
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
    x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    x = self.pooling(x)
    if self.num_classes == 1:
      x = self.dropout(x, training=training)
      x = self.linear_layer(x)
      x = self.dropout(x, training=training)
      x = self.output_layer(x)

    return x

    # return x  # (batch_size, input_seq_len, embedding_dim)

  def get_config(self):
    return {
      "num_classes": self.num_classes,
      "embedding_dim": self.embedding_dim,
      "num_heads": self.num_heads,
      "ff_dim": self.ff_dim,
      "rate": self.rate,
      "num_layers": self.num_layers,
    }


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(embedding_dim, num_heads)
    self.ffn = point_wise_feed_forward_network(embedding_dim, ff_dim)

    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)

  def call(self, x, training=None, mask=None):
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embedding_dim)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_dim)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_dim)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_dim)

    return out2  # (batch_size, input_seq_len, embedding_dim)

  def get_config(self):
    return {
      "embedding_dim": self.embedding_dim,
      "num_heads": self.num_heads,
      "ff_dim": self.ff_dim,
      "rate": self.rate,
    }


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(embedding_dim, num_heads)
    self.mha2 = MultiHeadAttention(embedding_dim, num_heads)

    self.ffn = point_wise_feed_forward_network(embedding_dim, ff_dim)

    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)
    self.dropout3 = layers.Dropout(rate)

  def call(self, x, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
    # enc_output.shape == (batch_size, input_seq_len, embedding_dim)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, embedding_dim)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
      enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, embedding_dim)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, embedding_dim)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embedding_dim)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embedding_dim)

    return out3, attn_weights_block1, attn_weights_block2

  def get_config(self):
    return {
      "embedding_dim": self.embedding_dim,
      "num_heads": self.num_heads,
      "ff_dim": self.ff_dim,
      "rate": self.rate,
    }


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.embedding_dim = embedding_dim

    assert embedding_dim % self.num_heads == 0

    self.depth = embedding_dim // self.num_heads

    self.wq = layers.Dense(embedding_dim)
    self.wk = layers.Dense(embedding_dim)
    self.wv = layers.Dense(embedding_dim)

    self.dense = layers.Dense(embedding_dim)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, embedding_dim)
    k = self.wk(k)  # (batch_size, seq_len, embedding_dim)
    v = self.wv(v)  # (batch_size, seq_len, embedding_dim)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.embedding_dim))  # (batch_size, seq_len_q, embedding_dim)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, embedding_dim)

    return output, attention_weights  # (batch_size, seq_len_q, embedding_dim), (batch_size, num_heads, seq_len_q, seq_len_k)

  def get_config(self):
    return {
      "embedding_dim": self.embedding_dim,
      "num_heads": self.num_heads,
    }


def positional_encoding(position, embedding_dim):
  pos_encoding = np.array(
    [[pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)] for pos in range(position)])
  pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
  pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
  pos_encoding = pos_encoding[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)  # (1, position, embedding_dim)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights  # (..., seq_len_q, depth_v), (..., seq_len_q, seq_len_k)


def point_wise_feed_forward_network(embedding_dim, ff_dim):
  return tf.keras.Sequential([
    layers.Dense(ff_dim, activation='relu'),  # (batch_size, seq_len, ff_dim)
    layers.Dense(embedding_dim)  # (batch_size, seq_len, embedding_dim)
  ])


if __name__ == '__main__':
  # test positional encoding
  pos_encoding = positional_encoding(50, 512)
  print(pos_encoding.shape)

  # test scaled dot product attention
  temp_k = tf.constant([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]], dtype=tf.float32)  # (4, 3)

  temp_v = tf.constant([[1, 0],
                        [10, 0],
                        [100, 5],
                        [1000, 6]], dtype=tf.float32)  # (4, 2)

  # This `query` aligns with the second `key
  # This will be the second output below
  temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

  temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
  print(temp_out)
  print(temp_attn)

  # This query aligns with a repeated key (third and fourth).
  # Because of the mask the second output is ignored.
  temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
  temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
  print(temp_out)
  print(temp_attn)

  # Test transformer input layer (str, numpy, list, tf.Tensor)
  transformer = TextTransformer(64, 4, 512, 4, 0.1, num_classes=None)
  print(transformer("This is a test"))
  print(transformer(np.array(['hello world', 'hello world'])))
  print(transformer(['hello world', 'hello world']))
  print(transformer(transformer.tokenizer.encode('hello world')))
  print(transformer(tf.constant(['hello world', 'hello world'])))


