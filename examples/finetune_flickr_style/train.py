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
from apollo.models import alexnet

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
    
    alexnet_layers = alexnet.alexnet_layers()
    loss = 0.
    for layer in alexnet_layers:
        loss += net.forward_layer(layer)

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
