import sys
import os
import logging
import argparse
import random
import numpy as np

import apollo

def default_hyper():
    hyper = {}
    hyper['gpu'] = None
    hyper['display_interval'] = 100
    hyper['random_seed'] = 91
    hyper['max_iter'] = sys.maxint
    hyper['test_interval'] = None
    hyper['test_iter'] = 1
    hyper['snapshot_interval'] = 10000
    hyper['snapshot_prefix'] = "/tmp"
    hyper['graph_interval'] = 500
    hyper['graph_prefix'] = '/tmp'
    hyper['schematic_prefix'] = '/tmp'
    return hyper

def validate_hyper(hyper):
    if 'base_lr' not in hyper:
        raise AttributeError('hyper is missing base_lr')
    if not os.path.isdir(hyper['snapshot_prefix']):
        raise OSError('%s does not exist' % hyper['snapshot_prefix'])
    if not os.path.isdir(hyper['graph_prefix']):
        raise OSError('%s does not exist' % hyper['graph_prefix'])
    if not os.path.isdir(hyper['schematic_prefix']):
        raise OSError('%s does not exist' % hyper['schematic_prefix'])

def default_train(hyper, forward, test_forward=None):
    if test_forward is None:
        test_forward=forward
    import matplotlib; matplotlib.use('Agg', warn=False); import matplotlib.pyplot as plt
    d = default_hyper()
    d.update(hyper)
    hyper = d
    validate_hyper(hyper)
    apollo.set_random_seed(hyper['random_seed'])
    if hyper['gpu'] is not None:
        apollo.Caffe.set_mode_gpu()
        apollo.Caffe.set_device(hyper['gpu'])
    else:
        apollo.Caffe.set_mode_cpu()
    apollo.Caffe.set_logging_verbosity(hyper['loglevel'])

    if hyper['gpu'] is None:
        logging.info('Using cpu device (pass --gpu X to train on the gpu)')
    else:
        logging.info('Using gpu device %d' % hyper['gpu'])
    net = apollo.Net()
    forward(net, hyper)
    network_path = '%s/network.jpg' % hyper['schematic_prefix']
    net.draw_to_file(network_path)
    logging.info('Drawing network to %s' % network_path)
    net.reset_forward()

    if hyper.get('separate_test_net', True) == True:
        test_net = apollo.Net()
        test_forward(test_net, hyper)
        test_net.reset_forward()
    else:
        test_net = net
    if 'weights' in hyper:
        net.load(hyper['weights'])

    train_loss_hist = []
    for i in xrange(hyper['start_iter'], hyper['max_iter']):
        train_loss_hist.append(forward(net, hyper))
        net.backward()
        lr = (hyper['base_lr'] * hyper.get('gamma', 1.)**(i // hyper.get('stepsize', sys.maxint)))
        net.update(lr=lr, momentum=hyper.get('momentum', 0.0),
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper.get('weight_decay', 0.0))
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        if i % hyper['snapshot_interval'] == 0 and i > hyper['start_iter']:
            filename = '%s/%d.h5' % (hyper['snapshot_prefix'], i)
            logging.info('Saving net to: %s' % filename)
            net.save(filename)
        if i % hyper['graph_interval'] == 0 and i > hyper['start_iter']:
            sub = hyper.get('sub', 100)
            plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
            filename = '%s/train_loss.jpg' % hyper['graph_prefix']
            logging.info('Saving figure to: %s' % filename)
            plt.savefig(filename)
        if hyper['test_interval'] is not None and i % hyper['test_interval'] == 0:
            test_loss = []
            accuracy = []
            test_net.phase = 'test'
            test_net.copy_params_from(net)
            for j in xrange(hyper['test_iter']):
                test_loss.append(test_forward(test_net, hyper))
                test_net.reset_forward()
                if 'accuracy' in test_net.tops:
                    accuracy.append(test_net.tops['accuracy'].data.flatten()[0])
            if len(accuracy) > 0:
                logging.info('Accuracy: %.5f' % np.mean(accuracy))
            logging.info('Test loss: %f' % np.mean(test_loss))
            test_net.phase = 'train'

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--loglevel', default=3, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    return parser
