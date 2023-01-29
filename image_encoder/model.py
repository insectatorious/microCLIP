# A simple ResNet model built with the Functional API
from typing import Optional, List, Dict

import tensorflow as tf
from keras.layers import LeakyReLU, Add, BatchNormalization, Conv2D
from tensorflow import Tensor


class ResNet_Auto(tf.keras.Model):
  def __init__(self,
               num_classes: Optional[int] = None,
               pooling: Optional[str] = "avg",
               name: Optional[str] = "resnet",
               # relu_type: Optional[str] = "relu",
               ):
    super(ResNet_Auto, self).__init__(name=name)
    if pooling not in {"avg", "max", None}:
      raise ValueError("pooling must be one of 'avg', 'max', or None.")

    # allowed_relu_types = {"relu", "leaky_relu", "elu", "selu", "swish", "mish", "gelu", "gelu_tanh"}
    # if relu_type not in allowed_relu_types:
    #   raise ValueError(f"relu_type must be one of {allowed_relu_types}.")

    self.num_classes = num_classes
    self.pooling = pooling
    # self.relu_type = relu_type
    # self.relu_fn = getattr(tf.nn, relu_type)

    self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()
    self.max_pool = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")

    self.block1 = ResNetBlock(64, 3, strides=1)
    self.block2 = ResNetBlock(128, 4, strides=2)
    self.block3 = ResNetBlock(256, 6, strides=2)
    self.block4 = ResNetBlock(512, 3, strides=2)

    if self.pooling == "avg":
      self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    elif self.pooling == "max":
      self.avg_pool = tf.keras.layers.GlobalMaxPool2D()
    else:
      self.avg_pool = None
    if self.num_classes:
      self.fc = tf.keras.layers.Dense(self.num_classes)
    else:
      self.fc = None

  def call(self, inputs, training=False, mask=None):
    # # Preprocess the input
    x = self.preprocess_input(inputs)
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.relu(x)
    x = self.max_pool(x)

    x = self.block1(x, training=training, mask=mask)
    x = self.block2(x, training=training, mask=mask)
    x = self.block3(x, training=training, mask=mask)
    x = self.block4(x, training=training, mask=mask)

    x = self.avg_pool(x)
    x = self.fc(x)

    return x

  def get_config(self):
    config = super(ResNet_Auto, self).get_config()
    config.update({"num_classes": self.num_classes, "pooling": self.pooling})
    return config

  @staticmethod
  def preprocess_input(x):
    x = tf.cast(x, tf.float32)
    x = x / 255.0
    # x = x - 0.5
    # x = x * 2.0
    return x


class ResNetBlock(tf.keras.layers.Layer):
  def __init__(self, filters, num_residuals, strides=1):
    super(ResNetBlock, self).__init__()
    self.filters = filters
    self.num_residuals = num_residuals
    self.strides = strides

    self.conv1 = tf.keras.layers.Conv2D(filters, 1, strides=strides)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv3 = tf.keras.layers.Conv2D(filters, 1, strides=1)
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.ReLU()
    self.sequential = tf.keras.Sequential()
    for _ in range(num_residuals - 1):
      self.sequential.add(ResidualUnit(filters, strides=1))
    self.sequential.add(ResidualUnit(filters, strides=strides))

  def call(self, inputs, training=False, mask=None):
    print(f"Processing block with {self.num_residuals} residual units")
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.relu(x)
    print("x1", x.shape)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.relu(x)
    print("x2", x.shape)
    x = self.conv3(x)
    x = self.bn3(x, training=training)
    print("x3", x.shape)

    residual = self.sequential(inputs, training=training)
    print("residual", residual.shape)
    print("x4", x.shape)
    x += residual
    x = self.relu(x)

    return x


class ResidualUnit(tf.keras.layers.Layer):
  def __init__(self, filters, strides=1):
    super(ResidualUnit, self).__init__()
    self.filters = filters
    self.strides = strides

    self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()

    if self.strides != 1 or True:
      self.conv3 = tf.keras.layers.Conv2D(filters, 1, strides=strides)
      self.bn3 = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs, training=False, mask=None):
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    residual = inputs
    # if self.strides != 1:
    residual = self.conv3(inputs)
    residual = self.bn3(residual, training=training)

    x += residual
    x = self.relu(x)

    return x


class ResNet(tf.keras.models.Model):

  @property
  def MEAN_RGB(self):
    return [0.6525933, 0.6365939, 0.61723816]

  def __init__(self,
               num_classes: Optional[int] = None,
               pooling: Optional[str] = "avg",
               name: Optional[str] = "resnet",
               pad_input: bool = False,
               filters: int = 64,
               dropout_rate: float = 0.1,
               # relu_type: Optional[str] = "relu",
               **kwargs):
    super(ResNet, self).__init__(name=name, **kwargs)

    if pooling not in {"avg", "max", None}:
      raise ValueError("pooling must be one of 'avg', 'max', or None.")

    self.num_classes = num_classes
    self.pooling = pooling
    self.pad_input = pad_input
    self.dropout_rate = dropout_rate

    self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same")
    self.spatial_dropout1 = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()
    self.max_pool = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")

    self.block1 = ResidualBlock(filters)
    self.spatial_dropout2 = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.block2 = ResidualBlock(filters)
    self.spatial_dropout3 = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.block3 = ResidualBlock(filters)
    self.spatial_dropout4 = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.block4 = ResidualBlock(filters)
    self.spatial_dropout5 = tf.keras.layers.SpatialDropout2D(dropout_rate)

    if self.pooling == "avg":
      self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    elif self.pooling == "max":
      self.avg_pool = tf.keras.layers.GlobalMaxPool2D()
    else:
      self.avg_pool = None

    if self.num_classes and not self.pooling:
      self.flatten = tf.keras.layers.Flatten()
    if self.num_classes:
      self.bottleneck = tf.keras.layers.Dense(512)
      self.fc = tf.keras.layers.Dense(self.num_classes)
    else:
      self.fc = None

  def call(self, inputs, training=False, mask=None):
    if self.pad_input:
      inputs = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]])

    x = self.conv1(inputs)
    x = self.spatial_dropout1(x, training=training)
    x = self.bn1(x, training=training)
    x = self.relu(x)
    x = self.max_pool(x)

    x = self.block1(x, training=training)
    x = self.spatial_dropout2(x, training=training)
    x = self.block2(x, training=training)
    x = self.spatial_dropout3(x, training=training)
    x = self.block3(x, training=training)
    x = self.spatial_dropout4(x, training=training)
    x = self.block4(x, training=training)
    x = self.spatial_dropout5(x, training=training)

    if self.pooling:
      x = self.avg_pool(x)
    if self.num_classes and not self.pooling:
      x = self.flatten(x)
    if self.num_classes:
      x = self.bottleneck(x)
      x = self.fc(x)

    return x

  def get_config(self):
    config = super(ResNet, self).get_config()
    config.update({
      "num_classes": self.num_classes,
      "pooling": self.pooling,
      "pad_input": self.pad_input,
      "dropout_rate": self.dropout_rate,
    })
    return config

  @classmethod
  def image_preprocessor(cls, x, **kwargs):
    # Cast to float32 and normalize to [0, 1]

    return tf.divide(tf.cast(x, tf.float32), 255.0) - cls.MEAN_RGB


class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self,
               num_of_filters: int = 16,
               **kwargs):
    super(ResidualBlock, self).__init__(**kwargs)
    self.num_of_filters = num_of_filters
    self.conv_1 = None
    self.conv_2 = None
    self.relu_1 = None
    self.relu_2 = None
    self.norm_1 = None
    self.norm_2 = None
    self.add_1 = None

  def build(self, input_shape: List) -> None:
    self.conv_1 = Conv2D(filters=self.num_of_filters,
                         kernel_size=(1, 1),
                         input_shape=input_shape)

    self.norm_1 = BatchNormalization()
    self.relu_1 = LeakyReLU()
    self.conv_2 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         padding="same")
    self.norm_2 = BatchNormalization()

    self.add_1 = Add()
    self.relu_2 = LeakyReLU()

  def call(self, inputs: Tensor, training=None, mask=None) -> Tensor:
    layer = self.norm_1(inputs, training=training)
    layer = self.relu_1(layer)
    layer = self.conv_1(layer)

    layer = self.norm_2(layer, training=training)
    layer = self.relu_2(layer)
    layer = self.conv_2(layer)
    layer = self.add_1([layer, inputs])

    return layer

  def get_config(self) -> Dict:
    config = super(ResidualBlock, self).get_config()
    config.update({"num_of_filters": self.num_of_filters})

    return config


def main():
  model = ResNet(num_classes=10, pooling="avg")
  model.build(input_shape=(None, 28, 28, 1))
  model.summary()

  print(model(tf.ones(shape=(1, 28, 28, 1))))

  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = x_train[..., tf.newaxis]
  x_test = x_test[..., tf.newaxis]

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

  model.fit(x_train, y_train, epochs=5)

  model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
  main()
