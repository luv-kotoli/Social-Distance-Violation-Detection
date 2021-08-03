#from __future__ import print_function
import os
import sys
#import re
#import glob
import h5py
#import skvideo.io
import numpy as np
from tqdm import tqdm

np.set_printoptions(precision=6)
fixed_seed = 999
np.random.seed(fixed_seed)  # Set seed for reproducibility
import tensorflow as tf

#print("Using keras {}".format(keras.__version__))

from tensorflow.keras.optimizers import SGD

#from datagen_unet import datagen_unet

from net_def_unet_ms import build_model_UNet_model_api as build_UNet
#from net_def_unet_ms import build_model_UNet_model_api as build_UNet
import cv2
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def build_model_load_weights(image_dim, model_dir, model_name):
    opt = SGD(
            lr=0.0001,
            decay=0.0001,
            momentum=0.9,
            nesterov=False,
            clipnorm=5,
            clipvalue=1)
    model = build_UNet(
        batch_size=1,
        patch_size=image_dim,
        optimizer=opt,
        output_ROI_mask=False,
    )
    
    model.load_weights(filepath=model_dir+model_name, by_name=True)
    return model





def main(exp_name):
    scaler_stability_factor = 1000

    model_dir = './models/{}/'.format(exp_name)
    model_name = 'best.h5'
    
    counting_results_name = 'counting_results/'
    h5_savename = 'counting_num_msunet.h5'

    if os.path.isdir(counting_results_name)==False:
        os.mkdir(counting_results_name)

    model = build_model_load_weights(image_dim=(288, 384, 3),
                                     model_dir=model_dir,
                                     model_name=model_name)  # projection/
    model.summary()
    print(model_name)
    #################################################################
    train_path0 = 'D:/PETS2009/violation_datasets/'
    #train_path0 = '/public/xiaoxuayu3/PETS2009/violation_datasets/'
    test_view1_1 = train_path0 + 'test/view1.h5'
    test_view2_1 = train_path0 + 'test/view2.h5'
    test_view3_1 = train_path0 + 'test/view3.h5'
    test_GP_1 = train_path0 + 'test/gp_violations.h5'
    
    h5file_test_GP = [test_GP_1]

    h5file_test_view1 = [test_view1_1]
    h5file_test_view2 = [test_view2_1]
    h5file_test_view3 = [test_view3_1]

    
    h5file_view1 = h5file_test_view1
    h5file_view2 = h5file_test_view2
    h5file_view3 = h5file_test_view3
    h5file_GP = h5file_test_GP

    # load the train or test data
    with h5py.File(h5file_view1[0], 'r') as f:
        images_i = f['color_images'][()]
        #density_maps_i = f['density_maps'][()]
        #dmp_h = density_maps_i.shape[1]
        #dmp_w = density_maps_i.shape[2]
        img_h = images_i.shape[1]
        img_w = images_i.shape[2]

    with h5py.File(h5file_GP[0], 'r') as f:
        density_maps_i = f['density_maps'][()]
        gdmp_h = density_maps_i.shape[1]
        gdmp_w = density_maps_i.shape[2]

    # predi

    count_view1_roi_GP = []
    count_view2_roi_GP = []
    count_view3_roi_GP = []
    count_gplane = []
    pred_dmap_gplane = []

    for j in range(1):

        # view 1
        #density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
        images1 = np.zeros([1, img_h, img_w, 3])

        h5file_view1_i = h5file_view1[j]
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['color_images'][()]
            #density_maps_i = f['density_maps'][()]
        #density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)

        #density_maps1 = density_maps1[1:, :, :, :]
        images1 = images1[1:, :, :, :]
        h1_test = images1
        #h1_dmaps_test = density_maps1

        # view 2
        #density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
        images2 = np.zeros([1, img_h, img_w, 3])

        h5file_view2_i = h5file_view2[j]
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['color_images'][()]
            #density_maps_i = f['density_maps'][()]
        #density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)

        #density_maps2 = density_maps2[1:, :, :, :]
        images2 = images2[1:, :, :, :]
        h2_test = images2
        #h2_dmaps_test = density_maps2

        # view 3
        #density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
        images3 = np.zeros([1, img_h, img_w, 3])
        h5file_view3_i = h5file_view3[j]
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['color_images'][()]
            #density_maps_i = f['density_maps'][()]
        #density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)
        #density_maps3 = density_maps3[1:, :, :, :]
        images3 = images3[1:, :, :, :]
        h3_test = images3
        #h3_dmaps_test = density_maps3

        # GP
        density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
        # images4 = np.asarray([])
        h5file_GP_i = h5file_GP[j]
        with h5py.File(h5file_GP_i, 'r') as f:
            # images_i = f['images'].value
            density_maps_i = f['segment_maps'][()]
            density_maps_i = np.expand_dims(density_maps_i,axis=-1)
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)
        # images3 = np.concatenate([images3, images_i], 0)
        density_maps4 = density_maps4[1:, :, :, :]
        h4_dmaps_test = density_maps4

        # depth ratio maps input
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 2 * 4
        # view 1
        view1_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_1_depth_image_halfHeight.npz')
        view1_image_depth = view1_image_depth.f.arr_0
        h = view1_image_depth.shape[0]
        w = view1_image_depth.shape[1]
        h_scale = h / scale_size
        w_scale = w / scale_size
        view1_image_depth_resized = cv2.resize(view1_image_depth, (int(w_scale), int(h_scale)))

        # set the center's scale of the image view1 as median of the all scales
        scale_center = np.median(scale_range)
        depth_center = view1_image_depth_resized[int(h_scale / 2), int(w_scale / 2)]
        view1_image_depth_resized_log2 = np.log2(view1_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view1_image_depth_resized_log2
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

        # view 2
        view2_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_2_depth_image_halfHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (int(w_scale), int(h_scale)))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_3_depth_image_halfHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (int(w_scale), int(h_scale)))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

        # GP mask:
        view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
        view1_gp_mask = view1_gp_mask.f.arr_0
        view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
        view2_gp_mask = view2_gp_mask.f.arr_0
        view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
        view3_gp_mask = view3_gp_mask.f.arr_0

        view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
        view_gp_mask = np.clip(view_gp_mask, 0, 1)
        # plt.imshow(view_gp_mask)
        view1_gp_mask2 = cv2.resize(view1_gp_mask, (610,710))
        view2_gp_mask2 = cv2.resize(view2_gp_mask, (610,710))
        view3_gp_mask2 = cv2.resize(view3_gp_mask, (610,710))
        view_gp_mask = cv2.resize(view_gp_mask, (610,710))

        count1 = []
        count2 = []
        count3 = []

        list_pred = list()
        pred_dmaps_list = []
        image_dim = None
        f_count = 0

        #plt.figure()

        for i in tqdm(range(h3_test.shape[0])):
            # i = 121

            frame1_s0 = h1_test[i:i + 1]
            frame2_s0 = h2_test[i:i + 1]
            frame3_s0 = h3_test[i:i + 1]

            #dmap1 = h1_dmaps_test[i:i + 1]
            #dmap2 = h2_dmaps_test[i:i + 1]
            #dmap3 = h3_dmaps_test[i:i + 1]
            #dmap4 = h4_dmaps_test[i:i + 1]

         

            pred_dmap = model.predict_on_batch([frame1_s0, frame2_s0, frame3_s0])

           
            pred_dmap_0 = pred_dmap#[3]
            #count4_pred_i = np.sum(pred_dmap_0.flatten()) / 1000
            pred_dmap_gplane.append(pred_dmap_0)

            
           

    with h5py.File(h5_savename, 'w') as f:
        f.create_dataset("pred_dmap_gplane", data=pred_dmap_gplane)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=str,
        default='cell',
        action="store")
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default='',
        action="store")
    args = parser.parse_args()
    main(exp_name=args.exp_name)
