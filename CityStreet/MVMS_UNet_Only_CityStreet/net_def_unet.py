from __future__ import print_function
from tensorflow.keras.layers import Multiply, Add, Concatenate, Dropout, Cropping2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, UpSampling2D, Reshape
from UpSampling_layer import UpSampling_layer
from feature_scale_fusion_layer_rbm import feature_scale_fusion_layer_rbm
from feature_scale_fusion_layer_learnScaleSel import feature_scale_fusion_layer
#from spatial_transformer import SpatialTransformer
from spatial_transformer_v2 import SpatialTransformer_2DTo2D_real as SpatialTransformer
import tensorflow as tf
#import cv2
#from tensorflow.keras.layers import Lambda
import numpy as np
from MyKerasLayers import AcrossChannelLRN
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

import tensorflow.keras as keras

def unet(base_weight_decay, input_shape):
    x = Input(shape=input_shape[1:])
    
    # Conv block 1
    conv1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_1_1'
    )(x)
    conv1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_1_2'
    )(conv1)
    pool1 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_1'
    )(conv1)

     # Conv block 2
    conv2 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_2_1'
    )(pool1)
    conv2 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_2_2'
    )(conv2)
    pool2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_2'
    )(conv2)

     # Conv block 3
    conv3 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_3_1'
    )(pool2)
    conv3 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_3_2'
    )(conv3)
    pool3 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_3'
    )(conv3)

    # Conv block 4
    conv4 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_4_1'
    )(pool3)
    conv4 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_4_2'
    )(conv4)
    pool4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_4'
    )(conv4)

    # Conv block 5
    conv5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1024,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_5_1'
    )(pool4)
    conv5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1024,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_5_2'
    )(conv5)
    

    # conv block 6 with up-sampling
    up6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=512,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_6'
    )(UpSampling2D(size=(2,2))(conv5))
    up6 = Cropping2D(cropping=((0,0),(0,1)))(up6)
    merge6 = Concatenate(#name='merge_6',
            axis=3)([conv4,up6])

    conv6 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_6_1'
    )(merge6)
    conv6 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_6_2'
    )(conv6)

    # conv block 7 with up-sampling
    up7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=256,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_7'
    )(UpSampling2D(size=(2,2))(conv6))
    up7 = Cropping2D(cropping=((0,1),(0,1)))(up7)
    merge7 = Concatenate(#name='merge_7',
            axis=3)([conv3,up7])

    conv7 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_7_1'
    )(merge7)
    conv7 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_7_2'
    )(conv7)

    # conv block 8 with up-sampling
    up8 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=128,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_8'
    )(UpSampling2D(size=(2,2))(conv7))
    #up8 = Cropping2D(cropping=((0,1),(0,1)))(up8)
    merge8 = Concatenate(#name='merge_8',
        axis=3)([conv2,up8])

    conv8 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_8_1'
    )(merge8)
    conv8 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_8_2'
    )(conv8)

    # conv block 9 with up-sampling
    up9 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_9'
    )(UpSampling2D(size=(2,2))(conv8))
    merge9 = Concatenate(#name='merge_9',
        axis=3)([conv1,up9])

    conv9 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_9_1'
    )(merge9)
    conv9 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_9_2'
    )(conv9)
    conv9 =  Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='conv_9_3'
    )(conv9)

    return Model(x,conv9)

def build_model_UNet_model_api(batch_size,
                               optimizer,
                               patch_size=(128,128),
                               base_weight_decay=0.0005,
                               output_ROI_mask=True):
    net_name = 'Multi-view_UNet_Share'

    input_shape = (batch_size, patch_size[0],patch_size[1],3)

    input_v1 = Input(batch_shape=input_shape,name='input_v1')
    input_v2 = Input(batch_shape=input_shape,name='input_v2')
    input_v3 = Input(batch_shape=input_shape,name='input_v3')

    unet_share = unet(base_weight_decay,input_shape)

    output_1 = unet_share(input_v1)
    output_2 = unet_share(input_v2)
    output_3 = unet_share(input_v3)

    # project
    output_1 = SpatialTransformer(1, [768, 640])(output_1)
    output_2 = SpatialTransformer(2, [768, 640])(output_2)
    output_3 = SpatialTransformer(3, [768, 640])(output_3)

    output_1 = tf.squeeze(output_1,axis=0)
    output_2 = tf.squeeze(output_2,axis=0)
    output_3 = tf.squeeze(output_3,axis=0)

    # concatenate
    output = Concatenate()([output_1,output_2,output_3])

    # decode progress
    output = Conv2D(
            data_format='channels_last',
            trainable=True,
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(base_weight_decay),
            use_bias=True,
            activation='relu',
            #name='conv_9_3'
    )(output)
    output = Conv2D(
            data_format='channels_last',
            trainable=True,
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(base_weight_decay),
            use_bias=True,
            activation='relu',
            #name='conv_9_3'
    )(output)
    output = Conv2D(
            data_format='channels_last',
            trainable=True,
            filters=1,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(base_weight_decay),
            use_bias=True,
            activation='sigmoid',
            #name='conv_9_3'
    )(output)

    model = Model([input_v1,input_v2,input_v3],output,name=net_name)
    print('Compiling ...')
    model.compile(optimizer=optimizer, loss='mse')
    return model