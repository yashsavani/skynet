"""
See ./examples/finetune_flickr_style/readme.txt for explanation
"""

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import logging
import argparse
import random
import os

import apollo
from apollo import layers
apollo_root = os.environ['APOLLO_ROOT']

def get_hyper():
    hyper = {}
    hyper['net_prototxt'] = "models/finetune_flickr_style/train_val.prototxt"
    hyper['test_iter'] = 100
    hyper['test_interval'] = 100
    # lr for fine-tuning should be lower than when starting from scratch
    hyper['base_lr'] = 0.001
    hyper['lr_policy'] = "step"
    hyper['gamma'] = 0.1
    # stepsize should also be lower, as we're closer to being done
    hyper['stepsize'] = 20000
    hyper['display_interval'] = 20
    hyper['max_iter'] = 100000
    hyper['momentum'] = 0.9
    hyper['weight_decay'] = 0.0005
    hyper['snapshot_interval'] = 10000
    hyper['snapshot_prefix'] = "models/finetune_flickr_style/finetune_flickr_style"
    hyper['random_seed'] = 0
    hyper['graph_interval'] = 100
    hyper['graph_prefix'] = ''
    return hyper

hyper = get_hyper()

def forward(net):
    transform = layers.Transform(mirror=True, crop_size=227, mean_file="data/ilsvrc12/imagenet_mean.binaryproto")
    data = layers.ImageData(name="data", tops=["data", "label"], source="data/flickr_style/train.txt",
        batch_size=50, new_height=256, new_width=256, transform=transform)
    net.forward_layer(data)

    conv_weight_filler = layers.Filler(type="gaussian", std=0.01)
    bias_filler0 = layers.Filler(type="constant", value=0.0)
    bias_filler1 = layers.Filler(type="constant", value=1.0)
    conv_lr_mults = [1.0, 2.0]

    conv1 = layers.Convolution(name="conv1", bottoms=["data"], param_lr_mults=conv_lr_mults, kernel_size=11,
        stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96)
    relu1 = layers.ReLU(name="relu1", bottoms=["conv1"], tops=["conv1"])
    pool1 = layers.Pooling(name="pool1", bottoms=["conv1"], kernel_size=3, stride=2)
    lrn1 = layers.LRN(name="norm1", bottoms=["pool1"], local_size=5, alpha=0.0001, beta=0.75)
    net.forward_layer(conv1)
    net.forward_layer(relu1)
    net.forward_layer(pool1)
    net.forward_layer(lrn1)

    conv2 = layers.Convolution(name="conv2", bottoms=["norm1"], param_lr_mults=conv_lr_mults, kernel_size=5,
        pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256)
    relu2 = layers.ReLU(name="relu2", bottoms=["conv2"], tops=["conv2"])
    pool2 = layers.Pooling(name="pool2", bottoms=["conv2"], kernel_size=3, stride=2)
    lrn2 = layers.LRN(name="norm2", bottoms=["pool2"], local_size=5, alpha=0.0001, beta=0.75)
    net.forward_layer(conv2)
    net.forward_layer(relu2)
    net.forward_layer(pool2)
    net.forward_layer(lrn2)

    conv3 = layers.Convolution(name="conv3", bottoms=["norm2"], param_lr_mults=conv_lr_mults, kernel_size=3,
        pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384)
    relu3 = layers.ReLU(name="relu3", bottoms=["conv3"], tops=["conv3"])

    conv4 = layers.Convolution(name="conv4", bottoms=["conv3"], param_lr_mults=conv_lr_mults, kernel_size=3,
        pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384)
    relu4 = layers.ReLU(name="relu4", bottoms=["conv4"], tops=["conv4"])
    net.forward_layer(conv3)
    net.forward_layer(relu3)
    net.forward_layer(conv4)
    net.forward_layer(relu4)

    conv5 = layers.Convolution(name="conv5", bottoms=["conv4"], param_lr_mults=conv_lr_mults, kernel_size=3,
        pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256)
    relu5 = layers.ReLU(name="relu5", bottoms=["conv5"], tops=["conv5"])
    pool5 = layers.Pooling(name="pool5", bottoms=["conv5"], kernel_size=3, stride=2)
    net.forward_layer(conv5)
    net.forward_layer(relu5)
    net.forward_layer(pool5)

    fc6 = layers.InnerProduct(name="fc6", bottoms=["pool5"], param_lr_mults=conv_lr_mults,
        weight_filler=layers.Filler(type="gaussian", std=0.005),
        bias_filler=bias_filler1, num_output=4096)
    relu6 = layers.ReLU(name="relu6", bottoms=["fc6"], tops=["fc6"])
    drop6 = layers.Dropout(name="drop6", bottoms=["fc6"], tops=["fc6"], dropout_ratio=0.5, phase='TRAIN')
    net.forward_layer(fc6)
    net.forward_layer(relu6)
    net.forward_layer(drop6)

    fc7 = layers.InnerProduct(name="fc7", bottoms=["fc6"], param_lr_mults=[1.0, 2.0],
        weight_filler=layers.Filler(type="gaussian", std=0.005),
        bias_filler=bias_filler1, num_output=4096)
    relu7 = layers.ReLU(name="relu7", bottoms=["fc7"], tops=["fc7"])
    drop7 = layers.Dropout(name="drop7", bottoms=["fc7"], tops=["fc7"], dropout_ratio=0.5, phase='TRAIN')
    fc8 = layers.InnerProduct(name="fc8_flickr", bottoms=["fc7"], param_lr_mults=[10.0, 20.0],
        weight_filler=layers.Filler(type="gaussian", std=0.01),
        bias_filler=bias_filler0, num_output=20)
    net.forward_layer(fc7)
    net.forward_layer(relu7)
    net.forward_layer(drop7)
    net.forward_layer(fc8)

    softmax_loss = layers.SoftmaxWithLoss(name="loss", bottoms=["fc8_flickr", "label"])
    loss = net.forward_layer(softmax_loss)

    return loss

def train():
    net = apollo.Net()

    forward(net)
    net.reset_forward()
    net.load('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    train_loss_hist = []

    for i in range(hyper['max_iter']):
        train_loss_hist.append(forward(net))
        net.backward()
        lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        if i % hyper['snapshot_interval'] == 0 and i > 0:
            filename = '%s_%d.h5' % (hyper['snapshot_prefix'], i)
            logging.info('Saving net to: %s' % filename)
            net.save(filename)
            with open('/tmp/log.txt', 'w') as f:
                f.write(json.dumps(train_loss_hist))
        if i % hyper['graph_interval'] == 0 and i > 0:
            sub = 100
            plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
            filename = '%strain_loss.jpg' % hyper['graph_prefix']
            logging.info('Saving figure to: %s' % filename)
            plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--loglevel', default=3, type=int)
    args = parser.parse_args()
    random.seed(0)
    apollo.Caffe.set_random_seed(hyper['random_seed'])
    apollo.Caffe.set_mode_gpu()
    apollo.Caffe.set_device(args.gpu)
    apollo.Caffe.set_logging_verbosity(args.loglevel)

    train()

if __name__ == '__main__':
    main()
