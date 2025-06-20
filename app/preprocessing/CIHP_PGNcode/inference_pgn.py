# from __future__ import print_function
# import argparse
# from datetime import datetime
# import os
# import sys
# import time
# import scipy.misc
# import scipy.io as sio
# import cv2
# from glob import glob
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from utils import *

# N_CLASSES = 20

# # RESTORE_FROM = r'D:\assignments\fyp\final evaluation\CIHP_PGNcode\checkpoint\CIHP_pgn'

# base_dir = os.getcwd()
# RESTORE_FROM = os.path.join(base_dir, "CIHP_PGNcode", "checkpoint", "CIHP_pgn" , "inference_pgn.py")


# def main():
#     parser = argparse.ArgumentParser(description="Process a single image using PGN.")
#     parser.add_argument('--image-path', type=str, required=True, help='Path to the input image.')
#     args = parser.parse_args()
#     image_path = args.image_path
#     DATA_DIR = image_path
#     NUM_STEPS = len(os.listdir(DATA_DIR)) 
#     print(f"total test images: {NUM_STEPS}")
#     """Create the model and start the evaluation process."""
#     # Create queue coordinator.
#     coord = tf.train.Coordinator()
#     # Load reader.
#     with tf.name_scope("create_inputs"):
#         reader = TestImageReader(image_path, None, None, None, False, False, False, coord)
#         image = reader.image
#         image_rev = tf.reverse(image, tf.stack([1]))
#         image_list = reader.image_list 

#     image_batch = tf.stack([image, image_rev])
#     h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
#     image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
#     image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
#     image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
#     image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
#     image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))
         
#     # Create network.
#     with tf.variable_scope('', reuse=False):
#         net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
#     with tf.variable_scope('', reuse=True):
#         net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
#     with tf.variable_scope('', reuse=True):
#         net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
#     with tf.variable_scope('', reuse=True):
#         net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
#     with tf.variable_scope('', reuse=True):
#         net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
#     with tf.variable_scope('', reuse=True):
#         net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
#     # parsing net

#     parsing_out1_050 = net_050.layers['parsing_fc']
#     parsing_out1_075 = net_075.layers['parsing_fc']
#     parsing_out1_100 = net_100.layers['parsing_fc']
#     parsing_out1_125 = net_125.layers['parsing_fc']
#     parsing_out1_150 = net_150.layers['parsing_fc']
#     parsing_out1_175 = net_175.layers['parsing_fc']

#     parsing_out2_050 = net_050.layers['parsing_rf_fc']
#     parsing_out2_075 = net_075.layers['parsing_rf_fc']
#     parsing_out2_100 = net_100.layers['parsing_rf_fc']
#     parsing_out2_125 = net_125.layers['parsing_rf_fc']
#     parsing_out2_150 = net_150.layers['parsing_rf_fc']
#     parsing_out2_175 = net_175.layers['parsing_rf_fc']

#     # edge net
#     edge_out2_100 = net_100.layers['edge_rf_fc']
#     edge_out2_125 = net_125.layers['edge_rf_fc']
#     edge_out2_150 = net_150.layers['edge_rf_fc']
#     edge_out2_175 = net_175.layers['edge_rf_fc']


#     # combine resize
#     parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

#     parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
#                                             tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


#     edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
#     edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
#     edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
#     edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
#     edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)
                                           
#     raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
#     head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
#     tail_list = tf.unstack(tail_output, num=20, axis=2)
#     tail_list_rev = [None] * 20
#     for xx in range(14):
#         tail_list_rev[xx] = tail_list[xx]
#     tail_list_rev[14] = tail_list[15]
#     tail_list_rev[15] = tail_list[14]
#     tail_list_rev[16] = tail_list[17]
#     tail_list_rev[17] = tail_list[16]
#     tail_list_rev[18] = tail_list[19]
#     tail_list_rev[19] = tail_list[18]
#     tail_output_rev = tf.stack(tail_list_rev, axis=2)
#     tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
#     raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
#     raw_output_all = tf.expand_dims(raw_output_all, dim=0)
#     pred_scores = tf.reduce_max(raw_output_all, axis=3)
#     raw_output_all = tf.argmax(raw_output_all, axis=3)
#     pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.


#     raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
#     head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
#     tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
#     raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
#     raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
#     pred_edge = tf.sigmoid(raw_edge_all)
#     res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

#     # Which variables to load.
#     restore_var = tf.global_variables()
#     # Set up tf session and initialize variables. 
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     init = tf.global_variables_initializer()
    
#     sess.run(init)
#     sess.run(tf.local_variables_initializer())
    
#     # Load weights.

#     loader = tf.compat.v1.train.Saver(var_list=restore_var)
#     if RESTORE_FROM is not None:
#         if load(loader, sess, RESTORE_FROM):
#             print(" [*] Load SUCCESS")
#         else:
#             print(" [!] Load failed...")
#             return
    
#     # Start queue threads.
#     threads = tf.train.start_queue_runners(coord=coord, sess=sess)

#     # evaluate prosessing
#     # parsing_dir = r'D:\assignments\fyp\final evaluation\website\uploads\image-parse-v3'
    
#     base_dir = os.getcwd()
#     parsing_dir = os.path.join(base_dir, "uploads", "image-parse-v3")


#     if not os.path.exists(parsing_dir):
#         os.makedirs(parsing_dir)
#     edge_dir = './output/cihp_edge_maps'
#     if not os.path.exists(edge_dir):
#         os.makedirs(edge_dir)
#     # Iterate over training steps.
#     for step in range(NUM_STEPS):
#         print(step)
#         parsing_, scores, edge_ = sess.run([pred_all, pred_scores, pred_edge])
#         if step % 1 == 0:
#             print('step {:d}'.format(step))
#             print(image_list[step])
#         img_split = image_list[step].split('/')
#         img_id = img_split[-1][:-4]



#         cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])
#         # sio.savemat('{}/{}.mat'.format(parsing_dir, img_id), {'data': scores[0,:,:]})
#         cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0,:,:,0] * 255)
#         print("here")

#     coord.request_stop()
#     coord.join(threads)

# if __name__ == '__main__':

   
#     main()

# ##############################################################333


from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

N_CLASSES = 20

# Set path to model checkpoint
base_dir = os.getcwd()
RESTORE_FROM = os.path.join("preprocessing", "CIHP_PGNcode", "checkpoint", "CIHP_pgn")


def main():
    parser = argparse.ArgumentParser(description="Process a single image using PGN.")
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image directory.')
    args = parser.parse_args()
    
    image_path = args.image_path
    DATA_DIR = image_path
    NUM_STEPS = len(os.listdir(DATA_DIR)) 
    print(f"Total test images: {NUM_STEPS}")

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = TestImageReader(image_path, None, None, None, False, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, axis=[1])
        image_list = reader.image_list 

    image_batch = tf.stack([image, image_rev])
    h_orig = tf.to_float(tf.shape(image_batch)[1])
    w_orig = tf.to_float(tf.shape(image_batch)[2])

    def resize_image(scale):
        return tf.image.resize_images(image_batch, 
            [tf.to_int32(tf.multiply(h_orig, scale)), tf.to_int32(tf.multiply(w_orig, scale))])

    image_batch050 = resize_image(0.50)
    image_batch075 = resize_image(0.75)
    image_batch125 = resize_image(1.25)
    image_batch150 = resize_image(1.50)
    image_batch175 = resize_image(1.75)
         
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)

    parsing_outputs = [
        net_050.layers['parsing_fc'], net_075.layers['parsing_fc'], net_100.layers['parsing_fc'],
        net_125.layers['parsing_fc'], net_150.layers['parsing_fc'], net_175.layers['parsing_fc']
    ]

    parsing_resized = [tf.image.resize_images(p, tf.shape(image_batch)[1:3]) for p in parsing_outputs]
    parsing_out1 = tf.reduce_mean(tf.stack(parsing_resized), axis=0)

    edge_outs = [
        net_100.layers['edge_rf_fc'], net_125.layers['edge_rf_fc'], 
        net_150.layers['edge_rf_fc'], net_175.layers['edge_rf_fc']
    ]
    edge_resized = [tf.image.resize_images(e, tf.shape(image_batch)[1:3]) for e in edge_outs]
    edge_out2 = tf.reduce_mean(tf.stack(edge_resized), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)

    tail_output_rev = tf.reverse(tail_output, axis=[1])
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, axis=0)

    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, axis=3) 

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, axis=[1])
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, axis=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # Initialize session and load model
    restore_var = tf.global_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    parsing_dir = os.path.join(base_dir, "uploads", "image-parse-v3")
    edge_dir = os.path.join(base_dir, "output", "cihp_edge_maps")

    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    for step in range(NUM_STEPS):
        print(step)
        parsing_, scores, edge_ = sess.run([pred_all, pred_scores, pred_edge])
        if step % 1 == 0:
            print('Processing step {:d}'.format(step))
            print(image_list[step])
        
        img_id = os.path.splitext(os.path.basename(image_list[step]))[0]

        cv2.imwrite(f'{parsing_dir}/{img_id}.png', parsing_[0, :, :, 0])
        cv2.imwrite(f'{edge_dir}/{img_id}.png', edge_[0, :, :, 0] * 255)
        print("Processing completed.")

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
