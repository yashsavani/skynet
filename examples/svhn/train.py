"""
Script to train overfeat network and save 
parameters to disk.  
Use test.py to use these parameters to generate
predictions on test dataset.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import logging
import argparse
import random
import os
import json
import cv2
import h5py

import apollo
from overfeat_net import OverfeatNet
from load_svhn import get_batch_iterator, IMG_HEIGHT, IMG_WIDTH
from util import show_rect_list

# imports hyper global variable containing training hyperparameters
from hyper import hyper 

# --- Parse command line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--loglevel', default=3, type=int)
parser.add_argument('-p', '--parameter_file', default="", type=str, help="Parameter file to resume training")
parser.add_argument('-n', '--name', default="latest", type=str, help="Session name; prefix used to save snapshots and loss history")
args = parser.parse_args()
random.seed(0)

# --- Use them to init caffe state ---
apollo.Caffe.set_random_seed(hyper['random_seed'])
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(args.gpu)
apollo.Caffe.set_logging_verbosity(args.loglevel)

# --- Initialize network ---
net = apollo.Net()
batch_iter = get_batch_iterator(hyper['batch_size'])
test_batch_iter = get_batch_iterator(hyper['batch_size'], data_type = "test")

batch = batch_iter.next()
overfeat_net = OverfeatNet(net, batch)
if args.parameter_file:
    overfeat_net.net.load(args.parameter_file)

# --- Do the training --- 
train_loss_hist = []
binary_softmax_hist = []
label_softmax_hist = []
bbox_loss_hist = []

for i in range(hyper['max_iter']):
    # do forward/backward pass
    batch = batch_iter.next()
    (binary_softmax_loss, label_softmax_loss, bbox_loss) = overfeat_net.train_batch_is(batch)
    loss = binary_softmax_loss + label_softmax_loss + bbox_loss
    lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
    overfeat_net.param_update(lr)

    # --- Log/Save/Graph progres ---
    train_loss_hist.append(loss)
    binary_softmax_hist.append(binary_softmax_loss)
    label_softmax_hist.append(label_softmax_loss)
    bbox_loss_hist.append(bbox_loss)

    if i % hyper['test_interval'] == 0: 
        #for rect_list in overfeat_net.rect_list_iter():
        test_batch = test_batch_iter.next()
        overfeat_net.train_batch_is(test_batch)
        overfeat_net.net.reset_forward()
        for idx in xrange(overfeat_net.batch_size()):
            rect_list = overfeat_net.rect_list(idx)
            if len(rect_list) > 0:
                print "made detections"
                rect_list.sort(key = lambda x: x.xmin)
                rect_str_list = [str(int(x.label) % 10) for x in rect_list]
                label_str =  "".join(rect_str_list)
            else:
                label_str = ""

            datum = test_batch.datum_list[idx]
            image_new = show_rect_list(datum.image, rect_list)

            # draw labels on image (sorted by xmin coordinate)
            if label_str:
                (h, w, _) = image_new.shape
                xpos = w / 4
                ypos = h / 4
                cv2.putText(image_new, label_str, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            path = "test/%d.png" %(idx)
            cv2.imwrite(path, image_new)
            logging.info("Saved to: " + path)

    if i % hyper['display_interval'] == 0:
        logging.info('Iteration %d loss: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        logging.info('Iteration %d binary softmax loss: %s' % (i, np.mean(binary_softmax_hist[-hyper['display_interval']:])))
        logging.info('Iteration %d label softmax loss: %s' % (i, np.mean(label_softmax_hist[-hyper['display_interval']:])))
        logging.info('Iteration %d bounding box loss: %s' % (i, np.mean(bbox_loss_hist[-hyper['display_interval']:])))

    if i % hyper['snapshot_interval'] == 0 and i > 0:
        filename = args.name + "Param.h5"
        logging.info('Saving net to: %s' % filename)
        net.save(filename)

    if i % hyper['graph_interval'] == 0 and i > 0:
        sub = hyper['display_interval']
        plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub], label = "total loss")
        plt.plot(np.convolve(bbox_loss_hist, np.ones(sub)/sub)[sub:-sub], label = "bbox loss")
        plt.plot(np.convolve(binary_softmax_hist, np.ones(sub)/sub)[sub:-sub], label = "binary label loss")
        plt.plot(np.convolve(label_softmax_hist, np.ones(sub)/sub)[sub:-sub], label = "label loss")
        plt.legend()
        filename = '%s_train_loss.jpg' % args.name
        logging.info('Saving figure to: %s' % filename)
        plt.savefig(filename)
        plt.clf()
