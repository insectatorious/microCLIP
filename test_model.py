import numpy as np
import tensorflow as tf

from image_encoder.model import ResNet
from model import MicroCLIP
from text_transformer.model import TextTransformer

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

clip = MicroCLIP(image_encoder=image_encoder,
                 text_encoder=text_transformer,
                 temperature=5.1)

clip((texts_tokenised, images))
clip.load_weights('clip.h5')

# Predict
predictions = clip.predict([texts_tokenised, images])
print(predictions)

predictions_softmax = tf.nn.softmax(predictions / clip.temperature, axis=0)
print(predictions_softmax)
