from __future__ import print_function
from tensorflow.keras.layers import Multiply, Add, Concatenate, Dropout, Cropping2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, UpSampling2D, Reshape
from tensorflow.python.training.tracking import base
from UpSampling_layer import UpSampling_layer
from feature_scale_fusion_layer_rbm import feature_scale_fusion_layer_rbm
from feature_scale_fusion_layer_learnScaleSel import feature_scale_fusion_layer
#from spatial_transformer import SpatialTransformer
from spatial_transformer_v3 import SpatialTransformer_2DTo2D_real as SpatialTransformer
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

def unet_conv_block1(base_weight_decay, x):
    # Conv block 1
    conv1 = Conv2D(
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
        #name='conv_1_1'
    )(x)
    conv1 = AcrossChannelLRN(local_size=5, alpha=0.01, beta=0.75, k=1,)(conv1)
    conv1 = Conv2D(
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
        #name='conv_1_2'
    )(conv1)
    conv1 = AcrossChannelLRN(local_size=5, alpha=0.01, beta=0.75, k=1,)(conv1)

    return conv1

def unet_conv_block2(base_weight_decay,x):

    pool1 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_1'
    )(x)

     # Conv block 2
    conv2 = Conv2D(
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
        #name='conv_2_1'
    )(pool1)
    conv2 = AcrossChannelLRN(local_size=5, alpha=0.01, beta=0.75, k=1,)(conv2)
    
    conv2 = Conv2D(
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
        #name='conv_2_2'
    )(conv2)

    return conv2

def unet_conv_block3(base_weight_decay,x): 
    pool2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        #name='pool_2'
    )(x)

     # Conv block 3
    conv3 = Conv2D(
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
        #name='conv_3_1'
    )(pool2)
    conv3 = AcrossChannelLRN(local_size=5, alpha=0.01, beta=0.75, k=1,)(conv3)
    
    conv3 = Conv2D(
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
        #name='conv_3_2'
    )(conv3)
    conv3 = AcrossChannelLRN(local_size=5, alpha=0.01, beta=0.75, k=1,)(conv3)

    return conv3

def unet_upsample_block4(base_weight_decay,x,concat):

    # conv block 4 with up-sampling
    up4 = Conv2D(
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
        #name='up_8'
    )(UpSampling2D(size=(2,2))(x))

    #up8 = Cropping2D(cropping=((0,1),(0,1)))(up8)
    merge4 = Concatenate(name='merge_4',)([concat,up4])

    conv4 =  Conv2D(
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
        #name='conv_8_1'
    )(merge4)
    conv4 =  Conv2D(
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
        #name='conv_8_2'
    )(conv4)

    return conv4

def unet_upsample_block5(base_weight_decay,x,concat):
    # conv block 9 with up-sampling
    up5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_9'
    )(UpSampling2D(size=(2,2))(x))
    merge5 = Concatenate(name='merge_5')([concat,up5])

    conv5 =  Conv2D(
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
        #name='conv_9_1'
    )(merge5)
    conv5 =  Conv2D(
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
        #name='conv_9_2'
    )(conv5)

    return conv5

def build_model_UNet_model_api(batch_size,
                               optimizer,
                               patch_size=(128,128),
                               base_weight_decay=0.0005,
                               output_ROI_mask=True):
    net_name = 'Multi-view_UNet_Share'

    # feature extracition blocks

    input_shape = (batch_size, patch_size[0],patch_size[1],3)

    input_v1 = Input(batch_shape=input_shape,name='input_v1')
    input_v2 = Input(batch_shape=input_shape,name='input_v2')
    input_v3 = Input(batch_shape=input_shape,name='input_v3')

    conv1_1 = unet_conv_block1(base_weight_decay,input_v1)
    conv1_2 = unet_conv_block1(base_weight_decay,input_v2)
    conv1_3 = unet_conv_block1(base_weight_decay,input_v3)

    conv2_1 = unet_conv_block2(base_weight_decay,conv1_1)
    conv2_2 = unet_conv_block2(base_weight_decay,conv1_2)
    conv2_3 = unet_conv_block2(base_weight_decay,conv1_3)

    conv3_1 = unet_conv_block3(base_weight_decay,conv2_1)
    conv3_2 = unet_conv_block3(base_weight_decay,conv2_2)
    conv3_3 = unet_conv_block3(base_weight_decay,conv2_3)

    # project scale 1
    proj1_1 = SpatialTransformer(1,[768, 640],scale=1)(conv1_1)
    proj1_2 = SpatialTransformer(2,[768, 640],scale=1)(conv1_2)
    proj1_3 = SpatialTransformer(3,[768, 640],scale=1)(conv1_3)

    # fusion
    concat_1 = Concatenate(name = 'concat_proj_1')([proj1_1,proj1_2,proj1_3])
    concat_1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        #name='up_8'
    )(concat_1)

    # project scale 2
    proj2_1 = SpatialTransformer(1,[768//2, 640//2],scale=2)(conv2_1)
    proj2_2 = SpatialTransformer(2,[768//2, 640//2],scale=2)(conv2_2)
    proj2_3 = SpatialTransformer(3,[768//2, 640//2],scale=2)(conv2_3)

    # fusion
    concat_2 = Concatenate(name = 'concat_proj_2')([proj2_1,proj2_2,proj2_3])
    concat_2 = Conv2D(
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
        #name='up_8'
    )(concat_2)

    # project scale 2
    proj3_1 = SpatialTransformer(1,[768//4, 640//4],scale=3)(conv3_1)
    proj3_2 = SpatialTransformer(2,[768//4, 640//4],scale=3)(conv3_2)
    proj3_3 = SpatialTransformer(3,[768//4, 640//4],scale=3)(conv3_3)

    # fusion
    concat_3 = Concatenate(name = 'concat_proj_3')([proj3_1,proj3_2,proj3_3])
    concat_3 = Conv2D(
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
    )(concat_3)

    conv4 = unet_upsample_block4(base_weight_decay,concat_3,concat_2)
    conv5 = unet_upsample_block5(base_weight_decay,conv4,concat_1)

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
    )(conv5)

    model = Model([input_v1,input_v2,input_v3],output,name=net_name)
    print('Compiling ...')
    model.compile(optimizer=optimizer, loss='mse')
    return model