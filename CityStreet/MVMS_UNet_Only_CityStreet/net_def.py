from __future__ import print_function
from tensorflow.keras.layers import Multiply, Add, Concatenate, Dropout
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, UpSampling2D, Reshape
from UpSampling_layer import UpSampling_layer
from feature_scale_fusion_layer_rbm import feature_scale_fusion_layer_rbm
from feature_scale_fusion_layer_learnScaleSel import feature_scale_fusion_layer
from spatial_transformer import SpatialTransformer
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Lambda
import numpy as np
from MyKerasLayers import AcrossChannelLRN
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

import tensorflow.keras as keras
assert keras.__version__.startswith('2.')
#from keras.layers.merge import Concatenate


# import sys
# sys.path.append('utils')
# from utils.data_utils import img_to_array, array_to_img
# from transformer import spatial_transformer_network


# feature extraction
def feature_extraction_view1(base_weight_decay, x):
    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x1_2)
    x1_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x1_2)
    x1_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x1_4)
    x1_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x1_4)

    # # conv block 5
    # x1_5 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=64,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_5'
    # )(x1_4)
    # x1_5 = Activation('relu',
    #                   #name='conv_block_5_act'
    #                   )(x1_5)
    #
    # # conv block 6
    # x1_6 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=32,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_6'
    # )(x1_5)
    # x1_6 = Activation('relu',
    #                   #name='conv_block_6_act'
    #                   )(x1_6)
    #
    # # conv block 7
    # x1_7 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=1,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_7'
    # )(x1_6)
    # x1_output = Activation('relu',
    #                        #name='conv_block_7_act'
    #                        )(x1_7)

    return x1_4


def feature_extraction_view2(base_weight_decay, x):
    x2_1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x2_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x2_1)
    x2_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x2_1)

    # conv block 2
    x2_2 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x2_1)
    x2_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x2_2)
    x2_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x2_2)
    x2_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x2_2)

    # conv block 3
    x2_3 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x2_2)
    x2_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x2_3)
    x2_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x2_3)

    # conv block 4
    x2_4 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x2_3)
    x2_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x2_4)
    x2_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x2_4)
    x2_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x2_4)

    # # conv block 5
    # x1_5 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=64,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_5'
    # )(x1_4)
    # x1_5 = Activation('relu',
    #                   #name='conv_block_5_act'
    #                   )(x1_5)
    #
    # # conv block 6
    # x1_6 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=32,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_6'
    # )(x1_5)
    # x1_6 = Activation('relu',
    #                   #name='conv_block_6_act'
    #                   )(x1_6)
    #
    # # conv block 7
    # x1_7 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=1,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_7'
    # )(x1_6)
    # x1_output = Activation('relu',
    #                        #name='conv_block_7_act'
    #                        )(x1_7)

    return x2_4


def feature_extraction_view3(base_weight_decay, x):
    x3_1 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x3_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x3_1)
    x3_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x3_1)

    # conv block 2
    x3_2 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x3_1)
    x3_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x3_2)
    x3_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x3_2)
    x3_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x3_2)

    # conv block 3
    x3_3 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x3_2)
    x3_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x3_3)
    x3_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x3_3)

    # conv block 4
    x3_4 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x3_3)
    x3_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x3_4)
    x3_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x3_4)
    x3_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x3_4)

    # # conv block 5
    # x1_5 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=64,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_5'
    # )(x1_4)
    # x1_5 = Activation('relu',
    #                   #name='conv_block_5_act'
    #                   )(x1_5)
    #
    # # conv block 6
    # x1_6 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=32,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_6'
    # )(x1_5)
    # x1_6 = Activation('relu',
    #                   #name='conv_block_6_act'
    #                   )(x1_6)
    #
    # # conv block 7
    # x1_7 = Conv2D(
    #     data_format='channels_last',
    #     trainable=True,
    #     filters=1,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     kernel_initializer='he_normal',
    #     padding='same',
    #     kernel_regularizer=l2(base_weight_decay),
    #     use_bias=True,
    #     activation=None,
    #     #name='conv_block_7'
    # )(x1_6)
    # x1_output = Activation('relu',
    #                        #name='conv_block_7_act'
    #                        )(x1_7)

    return x3_4

# single-view dmap output


def view1_decoder(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        # name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      # , name='conv_block_5_act'
                      )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      # , name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      # , name='conv_block_7_act'
                      )(x1_7)
    return x1_7


def view2_decoder(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        # name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      # , name='conv_block_5_act'
                      )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      # , name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      # , name='conv_block_7_act'
                      )(x1_7)
    return x1_7


def view3_decoder(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        # name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      # , name='conv_block_5_act'
                      )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      # , name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      # , name='conv_block_7_act'
                      )(x1_7)
    return x1_7

# fusion conv


def fusion_conv_v1(base_weight_decay, x):
    x1_02 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_02 = Activation('relu')(x1_02)
    return x1_02


def fusion_conv_v2(base_weight_decay, x):
    x1_02 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_02 = Activation('relu')(x1_02)
    return x1_02


def fusion_conv_v3(base_weight_decay, x):
    x1_03 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_03 = Activation('relu')(x1_03)
    return x1_03


######################## main structure #######################################

def scale_selection_mask(base_weight_decay, input_depth_maps):
    view1_scale = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=Constant(value=-1),
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        bias_initializer='ones',
        activation=None
        # name='scale_fusion1'
    )(input_depth_maps)
    #view1_scale_mask = Activation('softmax')(view1_scale)
    return view1_scale


def build_model_FCN_model_api(batch_size,
                              optimizer,
                              patch_size=(128, 128),
                              base_weight_decay=0.0005,
                              output_ROI_mask=True):
    print('Using build_model_FCN_model_api')

    # define the shared model:
    net_name = 'Multi-view_FCN'

    scale_number = 3
    ##################### input  ###############################################
    input_shape0 = (batch_size, int(patch_size[0]),   int(patch_size[1]),   3)
    input_shape1 = (batch_size, int(patch_size[0]/2), int(patch_size[1]/2), 3)
    input_shape2 = (batch_size, int(patch_size[0]/4), int(patch_size[1]/4), 3)

    input_shape3 = (1, int(patch_size[0]/4), int(patch_size[1]/4), 1)

    input_patches1_s0 = Input(batch_shape=input_shape0, name='patches1_s0')
    input_patches1_s1 = Input(batch_shape=input_shape1, name='patches1_s1')
    input_patches1_s2 = Input(batch_shape=input_shape2, name='patches1_s2')

    input_patches2_s0 = Input(batch_shape=input_shape0, name='patches2_s0')
    input_patches2_s1 = Input(batch_shape=input_shape1, name='patches2_s1')
    input_patches2_s2 = Input(batch_shape=input_shape2, name='patches2_s2')

    input_patches3_s0 = Input(batch_shape=input_shape0, name='patches3_s0')
    input_patches3_s1 = Input(batch_shape=input_shape1, name='patches3_s1')
    input_patches3_s2 = Input(batch_shape=input_shape2, name='patches3_s2')

    input_depth_maps_v1 = Input(
        batch_shape=input_shape3, name='depth_ratio_v1')
    input_depth_maps_v2 = Input(
        batch_shape=input_shape3, name='depth_ratio_v2')
    input_depth_maps_v3 = Input(
        batch_shape=input_shape3, name='depth_ratio_v3')

    if output_ROI_mask:
        # the output density patch/map is down-sampled by a factor of 4
        output_masks = Input(batch_shape=(batch_size, patch_size[0], patch_size[1], 3),
                             name='output_masks')

    ####################### view 1 #############################################
    # image pyramids:
    x1_s0_output = feature_extraction_view1(
        base_weight_decay, input_patches1_s0)
    x1_s1_output = feature_extraction_view1(
        base_weight_decay, input_patches1_s1)
    x1_s2_output = feature_extraction_view1(
        base_weight_decay, input_patches1_s2)

    # view 1 decoder
    # x1_7 = view1_decoder(base_weight_decay, x1_s0_output)

    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5')(x1_s0_output)
    x1_5 = Activation('relu', name='conv_block_5_act')(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6')(x1_5)
    x1_6 = Activation('relu', name='conv_block_6_act')(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7')(x1_6)
    x1_7 = Activation('relu', name='conv_block_7_act')(x1_7)

    ####################### view 2 #############################################
    # image pyramids:
    x2_s0_output = feature_extraction_view2(
        base_weight_decay, input_patches2_s0)
    x2_s1_output = feature_extraction_view2(
        base_weight_decay, input_patches2_s1)
    x2_s2_output = feature_extraction_view2(
        base_weight_decay, input_patches2_s2)

    #print('test', x2_s0_output.shape)
    # view 2 decoder
    # x2_7 = view2_decoder(base_weight_decay, x2_s0_output)

    # dmap
    # conv block 5
    x2_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_2')(x2_s0_output)
    x2_5 = Activation('relu', name='conv_block_5_2_act')(x2_5)

    # conv block 6
    x2_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_2')(x2_5)
    x2_6 = Activation('relu', name='conv_block_6_2_act')(x2_6)

    # conv block 7
    x2_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_2')(x2_6)
    x2_7 = Activation('relu', name='conv_block_7_2_act')(x2_7)

    ####################### view 3 #############################################
    # image pyramids:
    x3_s0_output = feature_extraction_view3(
        base_weight_decay, input_patches3_s0)
    x3_s1_output = feature_extraction_view3(
        base_weight_decay, input_patches3_s1)
    x3_s2_output = feature_extraction_view3(
        base_weight_decay, input_patches3_s2)

    # view 3 decoder
    # x3_7 = view3_decoder(base_weight_decay, x3_s0_output)

    # conv block 5
    x3_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_3'
    )(x3_s0_output)
    x3_5 = Activation('relu', name='conv_block_5_3_act')(x3_5)

    # conv block 6
    x3_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_3'
    )(x3_5)
    x3_6 = Activation('relu', name='conv_block_6_3_act')(x3_6)

    # conv block 7
    x3_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_3'
    )(x3_6)
    x3_7 = Activation('relu', name='conv_block_7_3_act')(x3_7)

    #################################### fusion #############################################
    ################# get the scale-selection mask #####################
    # view depth of image
    batch_size = x1_s0_output.shape[0]
    height = x1_s0_output.shape[1]
    width = x1_s0_output.shape[2]
    num_channels = x1_s0_output.shape[3]
    output_shape = [1, height, width, 1]

    # view1_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 1, output_shape=output_shape)
    # view2_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 2, output_shape=output_shape)
    # view3_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 3, output_shape=output_shape)

    view1_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v1)
    view2_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v2)
    view3_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v3)

    view1_scale_mask = feature_scale_fusion_layer_rbm(
        scale_number=scale_number)(view1_scale)
    view2_scale_mask = feature_scale_fusion_layer_rbm(
        scale_number=scale_number)(view2_scale)
    view3_scale_mask = feature_scale_fusion_layer_rbm(
        scale_number=scale_number)(view3_scale)

    #################### fusion with mask ##################
    # view 1
    # conv
    x1_s0_output_fusion = fusion_conv_v1(base_weight_decay, x1_s0_output)
    x1_s1_output_fusion = fusion_conv_v1(base_weight_decay, x1_s1_output)
    x1_s2_output_fusion = fusion_conv_v1(base_weight_decay, x1_s2_output)

    # up sampling
    x1_s1_output_fusion = UpSampling_layer(
        size=[height, width])([x1_s1_output_fusion])
    x1_s2_output_fusion = UpSampling_layer(
        size=[height, width])([x1_s2_output_fusion])

    concatenated_map_v1 = Concatenate(name='cat_map_v1')(
        [x1_s0_output_fusion, x1_s1_output_fusion, x1_s2_output_fusion])
    fusion_v1 = feature_scale_fusion_layer(scale_number=scale_number)(
        [concatenated_map_v1, view1_scale_mask])

    # proj
    fusion_v1_proj = SpatialTransformer(1, [int(768/4), int(640/4)])(fusion_v1)

    # view 2
    # conv
    x2_s0_output_fusion = fusion_conv_v2(base_weight_decay, x2_s0_output)
    x2_s1_output_fusion = fusion_conv_v2(base_weight_decay, x2_s1_output)
    x2_s2_output_fusion = fusion_conv_v2(base_weight_decay, x2_s2_output)

    # up sampling
    x2_s1_output_fusion = UpSampling_layer(
        size=[height, width])([x2_s1_output_fusion])
    x2_s2_output_fusion = UpSampling_layer(
        size=[height, width])([x2_s2_output_fusion])

    concatenated_map_v2 = Concatenate(name='cat_map_v2')(
        [x2_s0_output_fusion, x2_s1_output_fusion, x2_s2_output_fusion])
    fusion_v2 = feature_scale_fusion_layer(scale_number=scale_number)(
        [concatenated_map_v2, view2_scale_mask])

    # proj
    fusion_v2_proj = SpatialTransformer(2, [int(768/4), int(640/4)])(fusion_v2)

    # view 3
    # conv
    x3_s0_output_fusion = fusion_conv_v3(base_weight_decay, x3_s0_output)
    x3_s1_output_fusion = fusion_conv_v3(base_weight_decay, x3_s1_output)
    x3_s2_output_fusion = fusion_conv_v3(base_weight_decay, x3_s2_output)

    # up sampling
    x3_s1_output_fusion = UpSampling_layer(
        size=[height, width])([x3_s1_output_fusion])
    x3_s2_output_fusion = UpSampling_layer(
        size=[height, width])([x3_s2_output_fusion])

    concatenated_map_v3 = Concatenate(name='cat_map_v3')(
        [x3_s0_output_fusion, x3_s1_output_fusion, x3_s2_output_fusion])

    fusion_v3 = feature_scale_fusion_layer(scale_number=scale_number)(
        [concatenated_map_v3, view3_scale_mask])

    # proj
    fusion_v3_proj = SpatialTransformer(3, [int(768/4), int(640/4)])(fusion_v3)

    ################# concatenate ################
    concatenated_map = Concatenate(name='cat_map_fusion')(
        [fusion_v1_proj, fusion_v2_proj, fusion_v3_proj])
    fusion_v123 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=96,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='scale_fusion'
    )(concatenated_map)
    fusion_v123 = Activation('relu', name='scale_fusion_act')(fusion_v123)

    #################### fusion and decode #####################################
    # conv block 9
    '''
    x = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion1'
    )(fusion_v123)
    x = Activation('relu', name='conv_block_fusion1_act')(x)

    x = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion2'
    )(x)
    x = Activation('relu', name='conv_block_fusion2_act')(x)

    x = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion3'
    )(x)
    x_output = Activation('relu', name='conv_block_fusion3_act')(x)
    '''
    #######################################################################################

    ######################   U-Net for segmentation #######################################
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
        name='unet_conv1_1'
        )(fusion_v123)
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
        name='unet_conv1_2'
        )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
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
        name='unet_conv2_1'
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
        name='unet_conv2_2'
        )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
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
        name='unet_conv3_1'
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
        name='unet_conv3_2'
        )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
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
        name='unet_conv4_1'
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
        name='unet_conv4_2'
        )(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

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
        name='unet_conv5_1'
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
        name='unet_conv5_2'
        )(conv5)
    #drop5 = Dropout(0.5)(conv5)

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
        name='unet_up6'
        )(UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(name='unet_fusion_1',axis=3)([conv4, up6])
    conv6 = Conv2D(
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
        name='unet_conv6_1'
        )(merge6)
    conv6 = Conv2D(
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
        name='unet_conv6_2'
        )(conv6)

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
        name='unet_up7'
        )(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(name='unet_fusion_2',axis=3)([conv3, up7])
    conv7 = Conv2D(
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
        name='unet_conv7_1'
        )(merge7)
    conv7 = Conv2D(
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
        name='unet_conv7_2'
        )(conv7)

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
        name='unet_up8'
        )(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(name='unet_fusion_3',axis=3)([conv2, up8])
    conv8 = Conv2D(
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
        name='unet_conv8_1'
        )(merge8)
    conv8 = Conv2D(
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
        name='unet_conv8_2'
        )(conv8)

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
        name='unet_up9'
        )(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(name='unet_fusion_4',axis=3)([conv1, up9])
    conv9 = Conv2D(
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
        name='unet_conv9_1'
        )(merge9)
    conv9 = Conv2D(
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
        name='unet_conv9_2'
        )(conv9)
    conv9 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation='relu',
        name='unet_conv9_3'
        )(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid',name='unet_conv10')(conv9)

    #######################################################################################
    '''
    if output_ROI_mask:
        rgr_output = 'den_map_roi'
        output = Multiply(name=rgr_output)([x_output, output_masks])
        print('Layer name of regression output: {}'.format(rgr_output))
        model = Model(inputs=[input_patches1_s0, input_patches1_s1, input_patches1_s2,
                              input_patches2_s0, input_patches2_s1, input_patches2_s2,
                              input_patches3_s0, input_patches3_s1, input_patches3_s2,
                              input_depth_maps_v1, input_depth_maps_v2, input_depth_maps_v3,
                              output_masks],
                      outputs=[x_output],
                      name=net_name)
    else:
    '''
    model = Model(inputs=[input_patches1_s0, input_patches1_s1, input_patches1_s2,
                          input_patches2_s0, input_patches2_s1, input_patches2_s2,
                          input_patches3_s0, input_patches3_s1, input_patches3_s2,
                          input_depth_maps_v1, input_depth_maps_v2, input_depth_maps_v3],
                  outputs=[conv10],  # x1_7, x2_7, x3_7,
                  name=net_name+'overall')

    print('Compiling ...')
    model.compile(optimizer=optimizer, loss='mse')
    return model
