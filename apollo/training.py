import sys
import logging
import argparse
import random
import numpy as np

import apollo

def default_hyper(gpu=None,
        momentum=0.0,
        weight_decay=0.0,
        display_interval=100,
        clip_gradients=-1,
        stepsize=10000,
        gamma=1.0,
        random_seed=91,
        max_iter=sys.maxint,
        test_interval=None,
        test_iter=1,
        snapshot_interval=10000,
        snapshot_prefix="/tmp",
        graph_interval=500,
        graph_prefix='/tmp',
        schematic_prefix='/tmp'):
    hyper = {}
    hyper['gpu'] = gpu
    hyper['momentum'] = momentum
    hyper['weight_decay'] = weight_decay
    hyper['display_interval'] = display_interval
    hyper['max_iter'] = max_iter
    hyper['clip_gradients'] = clip_gradients
    hyper['snapshot_interval'] = snapshot_interval
    hyper['snapshot_prefix'] = snapshot_prefix
    hyper['stepsize'] = stepsize
    hyper['gamma'] = gamma
    hyper['random_seed'] = random_seed
    hyper['test_interval'] = test_interval
    hyper['test_iter'] = test_iter
    hyper['graph_interval'] = graph_interval
    hyper['graph_prefix'] = graph_prefix
    hyper['schematic_prefix'] = schematic_prefix
    return hyper

def validate_hyper(hyper):
    if 'base_lr' not in hyper:
        raise AttributeError('hyper is missing base_lr')

def train(hyper, forward, test_forward=None):
    if test_forward is None:
        test_forward=forward
    import matplotlib; matplotlib.use('Agg', warn=False); import matplotlib.pyplot as plt
    validate_hyper(hyper)
    apollo.init_flags(hyper)

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
        lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])
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

def init_flags(hyper):
    random.seed(hyper['random_seed'])
    np.random.seed(hyper['random_seed'])
    apollo.Caffe.set_random_seed(hyper['random_seed'])
    if hyper['gpu'] is not None:
        apollo.Caffe.set_mode_gpu()
        apollo.Caffe.set_device(hyper['gpu'])
    else:
        apollo.Caffe.set_mode_cpu()
    apollo.Caffe.set_logging_verbosity(hyper['loglevel'])

def update_hyper(hyper, args):
    for key,value in vars(args).iteritems():
        if value is not None:
            hyper[key] = value
