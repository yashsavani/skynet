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
import load_data
from apollo import layers
#from apollo.models import alexnet
apollo_root = os.environ['APOLLO_ROOT']

def get_hyper():
    hyper = {}
    hyper['test_iter'] = 100
    hyper['test_interval'] = 1000
    # lr for fine-tuning should be lower than when starting from scratch
    hyper['base_lr'] = 0.01
    hyper['lr_policy'] = "step"
    hyper['gamma'] = 0.1
    # stepsize should also be lower, as we're closer to being done
    hyper['stepsize'] = 20000
    hyper['display_interval'] = 10
    hyper['max_iter'] = 100000
    hyper['momentum'] = 0.9
    hyper['weight_decay'] = 0.0005
    hyper['snapshot_interval'] = 10000
    hyper['snapshot_prefix'] = "models/finetune_flickr_style/finetune_flickr_style"
    hyper['random_seed'] = 0
    hyper['graph_interval'] = 100
    hyper['graph_prefix'] = ''
    hyper['batch_size'] = 1
    return hyper

hyper = get_hyper()

def forward(net, batch_data):
    trans_img, bbox_label, conf_label = next(batch_data)

    net.forward_layer(layers.NumpyData(name="data", data=trans_img))
    net.forward_layer(layers.NumpyData(name="bbox_label", data=bbox_label))
    net.forward_layer(layers.NumpyData(name="conf_label", data=conf_label))
    
    alexnet_layers = alexnet.alexnet_layers()
    for layer in alexnet_layers:
        if layer.p.name == 'conv5':
            break
        net.forward_layer(layer)

    conv_weight_filler = layers.Filler(type="gaussian", std=0.01)
    bias_filler1 = layers.Filler(type="constant", value=1.0)
    conv_lr_mults = [1.0, 2.0]
    conv_decay_mults = [1.0, 0.0]

    net.forward_layer(layers.Convolution(name="L5", bottoms=["conv4"], param_lr_mults=conv_lr_mults,
        param_decay_mults=conv_decay_mults, kernel_size=3,
        pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1024))
    net.forward_layer(layers.ReLU(name="relu5", bottoms=["L5"], tops=["L5"]))
    net.forward_layer(layers.Pooling(name="pool5", bottoms=["L5"], kernel_size=3, stride=2, pad=1))

    net.forward_layer(layers.Convolution(name="L6", bottoms=["pool5"], param_lr_mults=conv_lr_mults,
        param_decay_mults=conv_decay_mults, kernel_size=1,
        weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1024))
    net.forward_layer(layers.ReLU(name="relu6", bottoms=["L6"], tops=["L6"]))
    net.forward_layer(layers.Dropout(name="drop6", bottoms=["L6"], tops=["L6"], dropout_ratio=0.5, phase='TRAIN'))

    net.forward_layer(layers.Convolution(name="L7", bottoms=["L6"], param_lr_mults=conv_lr_mults,
        param_decay_mults=conv_decay_mults, kernel_size=1,
        weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1024))
    net.forward_layer(layers.ReLU(name="relu7", bottoms=["L7"], tops=["L7"]))
    net.forward_layer(layers.Dropout(name="drop7", bottoms=["L7"], tops=["L7"], dropout_ratio=0.5, phase='TRAIN'))

    net.forward_layer(layers.Convolution(name="conf_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults,
        param_decay_mults=conv_decay_mults, kernel_size=1,
        weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=2))
    loss = 0.
    loss += net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss', bottoms=['conf_pred', 'conf_label']))
    net.forward_layer(layers.Softmax(name='softmax', bottoms=['conf_pred']))

    net.forward_layer(layers.Convolution(name="bbox_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults,
        param_decay_mults=conv_decay_mults, kernel_size=1,
        weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=4))

    net.forward_layer(layers.Concat(name='bbox_mask',
        bottoms=['conf_label', 'conf_label', 'conf_label', 'conf_label']))

    net.forward_layer(layers.Eltwise(name='bbox_pred_masked', bottoms=['bbox_pred', 'bbox_mask'], operation='PROD'))
    net.forward_layer(layers.Eltwise(name='bbox_label_masked', bottoms=['bbox_label', 'bbox_mask'], operation='PROD'))
    loss += net.forward_layer(layers.L1Loss(name='l1_loss', bottoms=['bbox_pred_masked', 'bbox_label_masked'], loss_weight=0.01))
    #print net.tops['conf_pred'].data[0,0]
    #print net.tops['softmax'].data[0,1] > 0.1
    #if counter > 100: import time; time.sleep(5)

    return loss

counter = 0
def train():
    net = apollo.Net()

    batch_data = load_data.get_data_batch_iter(hyper['batch_size'])

    forward(net, batch_data)
    net.reset_forward()

    #net.load(alexnet.weights_file())
    train_loss_hist = []

    for i in range(hyper['max_iter']):
        train_loss_hist.append(forward(net, batch_data))
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
