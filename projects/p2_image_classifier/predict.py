#imports
import argparse
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

# sample usage
# python predict.py ./test_images/orange_dahlia.jpg best_model.h5

# functions
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    adjusted_dimension_image = np.expand_dims(processed_test_image, axis=0)
    
    #make prediction
    predictions = model.predict(adjusted_dimension_image)
    first_predict = predictions[0]
    
    # sort and return classes and probs
    # ref. https://kite.com/python/answers/how-to-use-numpy-argsort-in-descending-order-in-python
    indices = np.argpartition(first_predict,-top_k)[-top_k:]
    sorted_indices = indices[np.argsort(first_predict[indices])]
    top_indices = sorted_indices[::-1][:len(sorted_indices)]
    top_indices_map = map(lambda x: str(x), top_indices)
    top_probs = first_predict[top_indices]
    top_probs_map = map(lambda x: '{0:.8f}'.format(x), top_probs)
    top_classes_map = map(lambda x: class_names[str(x+1)], top_indices)
    return top_probs, list(top_classes_map)

# handling arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', action="store", type=str)
parser.add_argument('model', action="store", type=str)
parser.add_argument('--category_names', action="store",dest="category_names", type=str)
parser.add_argument('--top_k', action="store",dest="top_k", type=int)
results = parser.parse_args()
i_path = results.path
i_model = results.model
i_category_names = results.category_names if (results.category_names) else "label_map.json" 
i_top_k = results.top_k if (results.top_k) else 5 
print('path     = {!r}'.format(i_path))
print('model     = {!r}'.format(i_model))
print('category_names     = {!r}'.format(i_category_names))
print('top_k     = {!r}'.format(i_top_k))

# load target model
reloaded_keras_model = tf.keras.models.load_model('./' + i_model, custom_objects={'KerasLayer':hub.KerasLayer})
#reloaded_keras_model.summary()

# load classname
with open(i_category_names, 'r') as f:
    class_names = json.load(f)

# predict
image_path = i_path
probs, classes = predict(image_path, reloaded_keras_model, i_top_k)
#print(probs)
print('Top', i_top_k, 'classes:', classes)

#print('test', time.time())