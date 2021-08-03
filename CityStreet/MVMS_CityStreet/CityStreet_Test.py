import os 
import sys
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from net_def import build_model_FCN_model_api as build_FCNN
from datagen_v3 import datagen_v3



import cv2
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set enough GPU memory as needed(default, all GPU memory is used)
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
    
    model.load_weights(filepath=model_dir+model_name,by_name=True)
    print(os.path.join(model_dir,model_name))
    return model

def main(exp_name):
    image_dim = (380, 676, 3)
    model_dir = './models/Street_all_output_{}/'.format(exp_name)
    model_name = 'best.h5'
    
    train_path0 = '../../violation_data/'
    test_view1_1 = train_path0 + 'test/view1_violation.h5'
    test_view2_1 = train_path0 + 'test/view2_violation.h5'
    test_view3_1 = train_path0 + 'test/view3_violation.h5'
    test_GP_1 = train_path0 + 'test/gp_violations.h5'
    
    h5file_test_GP = [test_GP_1]

    h5file_test_view1 = [test_view1_1]
    h5file_test_view2 = [test_view2_1]
    h5file_test_view3 = [test_view3_1]

    test_gen = datagen_v3(
        h5file_view1=h5file_test_view1,
        h5file_view2=h5file_test_view2,
        h5file_view3=h5file_test_view3,
        h5file_GP=h5file_test_GP,

        batch_size=1,
        images_per_set=None,
        patches_per_image=1,  # 1000,
        patch_dim=image_dim[:2],
        density_scaler=1000,
        image_shuffle=True,
        patch_shuffle=True,
        random_state=None
    )
    
    model = build_model_load_weights(image_dim,model_dir,model_name)
    
    result = model.predict(
        x= test_gen,
        batch_size=1,
        verbose=1,
        max_queue_size=20,
        workers=1,
    )
    
    return result


if __name__ == '__main__':
    main(exp_name='07210432')