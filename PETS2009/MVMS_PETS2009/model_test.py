import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cv2
import camera_proj_Tsai as proj
import tensorflow as tf

import h5py
#view1_h5 = '../../violation_data/train/view3_violation.h5'
view1_h5 = '/public/xiaoxuayu3/PETS2009/violation_datasets/train/view1.h5'
view1_data = h5py.File(view1_h5,'r')
print(view1_data.keys())

images = view1_data['color_images']

test_image = images[0]
print(test_image.shape)
print(test_image.dtype)

test_image = cv2.resize(test_image,(384//2,288//2))

#from spatial_transformer import SpatialTransformer
from spatial_transformer_v3 import SpatialTransformer_2DTo2D_real as SpatialTransformer
test_image = np.expand_dims(test_image,axis=0)

result = SpatialTransformer(1,[710,610],scale=2)(test_image)

print(result.shape)

result = tf.squeeze(result)
print(result)

with tf.compat.v1.Session() as sess:
    result = sess.run(result)
    
print(result.shape)

cv2.imwrite('image.jpg',result)