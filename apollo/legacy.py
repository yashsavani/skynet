#!/usr/bin/env python
import apollo
import imp
import numpy as np
import logging
import argparse

def run(hyper):
    net = apollo.Net()
    arch = apollo.Architecture()
    arch.load_from_proto(hyper['net_prototxt'])

    test_net = apollo.Net(phase='test')
    test_arch = apollo.Architecture(phase='test')
    test_arch.load_from_proto(hyper['net_prototxt'])

    arch.forward(net)
    if hyper['weights']:
        print 'loading'
        net.load(hyper['weights'])
    net.reset_forward()

    test_arch.forward(test_net)
    test_net.reset_forward()

    def test_performance(net, test_net):
        test_error = []
        test_net.copy_params_from(net)
        for _ in range(hyper['test_iter']):
            test_error.append(test_arch.forward(test_net))
            test_net.reset_forward()
        logging.info('Test Error: %f' % np.mean(test_error))

    error = []
    for i in range(hyper['max_iter']):
        error.append(arch.forward(net))
        net.backward()
        lr = (hyper['base_lr'] / (hyper.get('gamma', 1.))**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(error)))
            error = []
        if i % hyper['test_interval'] == 0:
            test_performance(net, test_net)
        if i % hyper['snapshot_interval'] == 0 and i > 0:
            net.save('%s_%d.h5' % (hyper['snapshot_prefix'], i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default=3)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--solver')
    args = parser.parse_args()

    config = imp.load_source('module.name', args.solver)
    hyper = config.get_hyper()
    hyper['weights'] = args.weights

    apollo.Caffe.set_random_seed(hyper.get('random_seed', 0))
    if args.gpu is not None:
        apollo.Caffe.set_mode_gpu()
        apollo.Caffe.set_device(int(args.gpu))
    else:
        apollo.Caffe.set_mode_cpu()
    apollo.Caffe.set_logging_verbosity(args.loglevel)
    run(hyper)


if __name__ == '__main__':
    main()
