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

from net_def import build_model_FCN_model_api as build_FCNN
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
    model = build_FCNN(
        batch_size=1,
        patch_size=image_dim,
        optimizer=opt,
        output_ROI_mask=False,
    )
    
    model.load_weights(filepath=model_dir+model_name, by_name=True)
    return model





def main(exp_name):
    scaler_stability_factor = 1000

    model_dir = './models/Street_all_output_{}/'.format(exp_name)
    model_name = 'best.h5'
    
    counting_results_name = 'counting_results/'
    h5_savename = counting_results_name + 'counting_num_{}.h5'.format(exp_name)

    if os.path.isdir(counting_results_name)==False:
        os.mkdir(counting_results_name)

    model = build_model_load_weights(image_dim=(380, 676, 3),
                                     model_dir=model_dir,
                                     model_name=model_name)  # projection/
    model.summary()
    print(model_name)
    #################################################################

    train_path0 = '../../violation_data/'
    test_view1_1 = train_path0 + 'test/view1_violation.h5'
    test_view2_1 = train_path0 + 'test/view2_violation.h5'
    test_view3_1 = train_path0 + 'test/view3_violation.h5'
    test_GP_1 = train_path0 + 'test/gp_violations_seg.h5'
    
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
        density_maps_i = f['density_maps'][()]
        dmp_h = density_maps_i.shape[1]
        dmp_w = density_maps_i.shape[2]
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
        density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
        images1 = np.zeros([1, img_h, img_w, 3])

        h5file_view1_i = h5file_view1[j]
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['color_images'][()]
            density_maps_i = f['density_maps'][()]
        density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)

        density_maps1 = density_maps1[1:, :, :, :]
        images1 = images1[1:, :, :, :]
        h1_test = images1
        h1_dmaps_test = density_maps1

        # view 2
        density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
        images2 = np.zeros([1, img_h, img_w, 3])

        h5file_view2_i = h5file_view2[j]
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['color_images'][()]
            density_maps_i = f['density_maps'][()]
        density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)

        density_maps2 = density_maps2[1:, :, :, :]
        images2 = images2[1:, :, :, :]
        h2_test = images2
        h2_dmaps_test = density_maps2

        # view 3
        density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
        images3 = np.zeros([1, img_h, img_w, 3])
        h5file_view3_i = h5file_view3[j]
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['color_images'][()]
            density_maps_i = f['density_maps'][()]
        density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)
        density_maps3 = density_maps3[1:, :, :, :]
        images3 = images3[1:, :, :, :]
        h3_test = images3
        h3_dmaps_test = density_maps3

        # GP
        density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
        # images4 = np.asarray([])
        h5file_GP_i = h5file_GP[j]
        with h5py.File(h5file_GP_i, 'r') as f:
            # images_i = f['images'].value
            density_maps_i = f['segment_maps'][()]
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)
        # images3 = np.concatenate([images3, images_i], 0)
        density_maps4 = density_maps4[1:, :, :, :]
        h4_dmaps_test = density_maps4

        # depth ratio maps input
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 4
        # view 1
        view1_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_1_depth_image_avgHeight.npz')
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
        view2_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_2_depth_image_avgHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (int(w_scale), int(h_scale)))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_3_depth_image_avgHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (int(w_scale), int(h_scale)))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

        # GP mask:
        view1_gp_mask = np.load('coords_correspondence_Street/mask/view1_GP_mask.npz')
        view1_gp_mask = view1_gp_mask.f.arr_0
        view2_gp_mask = np.load('coords_correspondence_Street/mask/view2_GP_mask.npz')
        view2_gp_mask = view2_gp_mask.f.arr_0
        view3_gp_mask = np.load('coords_correspondence_Street/mask/view3_GP_mask.npz')
        view3_gp_mask = view3_gp_mask.f.arr_0

        view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
        view_gp_mask = np.clip(view_gp_mask, 0, 1)
        # plt.imshow(view_gp_mask)
        view1_gp_mask2 = cv2.resize(view1_gp_mask, (640, 768))
        view2_gp_mask2 = cv2.resize(view2_gp_mask, (640, 768))
        view3_gp_mask2 = cv2.resize(view3_gp_mask, (640, 768))
        view_gp_mask = cv2.resize(view_gp_mask, (640, 768))

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
            frame1 = frame1_s0[0, :, :, :]

            frame1_s1_0 = cv2.resize(frame1, (frame1.shape[1] // 2, frame1.shape[0] // 2))
            frame1_s1 = np.expand_dims(frame1_s1_0, axis=0)
            #frame1_s1 = np.expand_dims(frame1_s1, axis=3)

            frame1_s2_0 = cv2.resize(frame1_s1_0, (frame1_s1_0.shape[1] // 2, frame1_s1_0.shape[0] // 2))
            frame1_s2 = np.expand_dims(frame1_s2_0, axis=0)
            #frame1_s2 = np.expand_dims(frame1_s2, axis=3)

            frame2_s0 = h2_test[i:i + 1]
            frame2 = frame2_s0[0, :, :, :]

            frame2_s1_0 = cv2.resize(frame2, (frame2.shape[1] // 2, frame2.shape[0] // 2))
            frame2_s1 = np.expand_dims(frame2_s1_0, axis=0)
            #frame2_s1 = np.expand_dims(frame2_s1, axis=3)

            frame2_s2_0 = cv2.resize(frame2_s1_0, (frame2_s1_0.shape[1] // 2, frame2_s1_0.shape[0] // 2))
            frame2_s2 = np.expand_dims(frame2_s2_0, axis=0)
            #frame2_s2 = np.expand_dims(frame2_s2, axis=3)

            frame3_s0 = h3_test[i:i + 1]
            frame3 = frame3_s0[0, :, :, :]

            frame3_s1_0 = cv2.resize(frame3, (frame3.shape[1] // 2, frame3.shape[0] // 2))
            frame3_s1 = np.expand_dims(frame3_s1_0, axis=0)
            #frame3_s1 = np.expand_dims(frame3_s1, axis=3)

            frame3_s2_0 = cv2.resize(frame3_s1_0, (frame3_s1_0.shape[1] // 2, frame3_s1_0.shape[0] // 2))
            frame3_s2 = np.expand_dims(frame3_s2_0, axis=0)
            #frame3_s2 = np.expand_dims(frame3_s2, axis=3)

            dmap1 = h1_dmaps_test[i:i + 1]
            dmap2 = h2_dmaps_test[i:i + 1]
            dmap3 = h3_dmaps_test[i:i + 1]
            dmap4 = h4_dmaps_test[i:i + 1]

            count1_gt_i = np.sum(np.sum(dmap1[0, :, :, 0]))
            count2_gt_i = np.sum(np.sum(dmap2[0, :, :, 0]))
            count3_gt_i = np.sum(np.sum(dmap3[0, :, :, 0]))
            count4_gt_i = np.sum(np.sum(dmap4[0, :, :, 0]))

            '''
            frame1_s0 = np.tile(frame1_s0, (1, 1, 1, 3))
            frame2_s0 = np.tile(frame2_s0, (1, 1, 1, 3))
            frame3_s0 = np.tile(frame3_s0, (1, 1, 1, 3))

            frame1_s1 = np.tile(frame1_s1, (1, 1, 1, 3))
            frame2_s1 = np.tile(frame2_s1, (1, 1, 1, 3))
            frame3_s1 = np.tile(frame3_s1, (1, 1, 1, 3))

            frame1_s2 = np.tile(frame1_s2, (1, 1, 1, 3))
            frame2_s2 = np.tile(frame2_s2, (1, 1, 1, 3))
            frame3_s2 = np.tile(frame3_s2, (1, 1, 1, 3))
            '''
            

            pred_dmap = model.predict_on_batch([frame1_s0, frame1_s1, frame1_s2,
                                                frame2_s0, frame2_s1, frame2_s2,
                                                frame3_s0, frame3_s1, frame3_s2,
                                                view1_image_depth_resized_log2,
                                                view2_image_depth_resized_log2,
                                                view3_image_depth_resized_log2])

            # count1_pred_i = np.sum(pred_dmap[0].flatten())/1000
            # count2_pred_i = np.sum(pred_dmap[1].flatten())/1000
            # count3_pred_i = np.sum(pred_dmap[2].flatten())/1000

            pred_dmap_0 = pred_dmap#[3]
            # pred_dmap_0 = pred_dmap_0*view_gp_mask
            count4_pred_i = np.sum(pred_dmap_0.flatten()) / 1000
            pred_dmap_gplane.append(pred_dmap_0)

            
            #count_gplane.append([count1_gt_i, count2_gt_i, count3_gt_i, count4_gt_i, count4_pred_i])
            '''
             # roi GP pred
            count_view1_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view1_gp_mask)) / 1000
            count_view2_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view2_gp_mask)) / 1000
            count_view3_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view3_gp_mask)) / 1000
            # roi GP gt
            count_view1_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view1_gp_mask2))
            count_view2_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view2_gp_mask2))
            count_view3_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view3_gp_mask2))
            count_view1_roi_GP.append([count_view1_roi_GP_gt_i, count_view1_roi_GP_i])
            count_view2_roi_GP.append([count_view2_roi_GP_gt_i, count_view2_roi_GP_i])
            count_view3_roi_GP.append([count_view3_roi_GP_gt_i, count_view3_roi_GP_i])

            '''
           

    #mae1 = np.asarray(count1)[:, 0] - np.asarray(count1)[:, 1]
    #mae1 = np.mean(np.abs(mae1))
    ##print('mae1: ',mae1)
    #mae2 = np.asarray(count2)[:, 0] - np.asarray(count2)[:, 1]
    #mae2 = np.mean(np.abs(mae2))
    #print('mae2: ',mae2)
    #mae3 = np.asarray(count3)[:, 0] - np.asarray(count3)[:, 1]
    #mae3 = np.mean(np.abs(mae3))
    #print('mae3:',mae3)

    '''
    # GP
    mae_GP = np.asarray(count_gplane)[:, 4] - np.asarray(count_gplane)[:, 3]
    mae_GP = np.mean(np.abs(mae_GP))
    print('mae_GP: ',mae_GP)

    # GP roi / GP
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view1_GP = np.mean(np.abs(dif_view1_GP))
    print(mae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view2_GP = np.mean(np.abs(dif_view2_GP))
    print(mae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view3_GP = np.mean(np.abs(dif_view3_GP))
    print(mae_view3_GP)

    # GP roi / GP roi
    mae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_view1_roi_GP)[:, 0]
    mae_view1_GProi = np.mean(np.abs(mae_view1_GProi))
    print(mae_view1_GProi)
    mae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_view2_roi_GP)[:, 0]
    mae_view2_GProi = np.mean(np.abs(mae_view2_GProi))
    print(mae_view2_GProi)
    mae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_view3_roi_GP)[:, 0]
    mae_view3_GProi = np.mean(np.abs(mae_view3_GProi))
    print(mae_view3_GProi)

    # GP roi/view
    dif_view1 = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 0]
    mae_view1 = np.mean(np.abs(dif_view1))
    print(mae_view1)
    dif_view2 = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 1]
    mae_view2 = np.mean(np.abs(dif_view2))
    print(mae_view2)
    dif_view3 = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 2]
    mae_view3 = np.mean(np.abs(dif_view3))
    print(mae_view3)
    '''
    

    with h5py.File(h5_savename, 'w') as f:
        #f.create_dataset("count1_GProi", data=count_view1_roi_GP)
        #f.create_dataset("count2_GProi", data=count_view2_roi_GP)
        #f.create_dataset("count3_GProi", data=count_view3_roi_GP)
        #f.create_dataset("count_gplane", data=count_gplane)
        #f.create_dataset("mae_GP", data=mae_GP)

        f.create_dataset("pred_dmap_gplane", data=pred_dmap_gplane)

        #f.create_dataset("mae_view1_GP", data=mae_view1_GP)
        #f.create_dataset("mae_view2_GP", data=mae_view2_GP)
        #f.create_dataset("mae_view3_GP", data=mae_view3_GP)

        #f.create_dataset("mae_view1", data=mae_view1)
        #f.create_dataset("mae_view2", data=mae_view2)
        #f.create_dataset("mae_view3", data=mae_view3)

        #f.create_dataset("mae_view1_GProi", data=mae_view1_GProi)
        #f.create_dataset("mae_view2_GProi", data=mae_view2_GProi)
        #f.create_dataset("mae_view3_GProi", data=mae_view3_GProi)




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
