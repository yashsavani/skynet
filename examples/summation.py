import logging
import numpy as np
import random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import argparse

import apollo
from apollo import layers

def forward(net, hyper):
    length = random.randrange(5, 15)

    # initialize all weights in [-0.1, 0.1]
    filler = layers.Filler(type='uniform', min=-hyper['init_range'],
        max=hyper['init_range'])
    # initialize the LSTM memory with all 0's
    net.forward_layer(layers.NumpyData(name='lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    accum = np.zeros((hyper['batch_size'],))

    # Begin recurrence through 5 - 15 inputs
    for step in range(length):
        # Set up the value blob
        net.forward_layer(layers.DummyData(name='value%d' % step,
            shape=[hyper['batch_size'], 1, 1, 1]))
        value = np.array([random.random() for _ in range(hyper['batch_size'])])
        accum += value
        # Set data of value blob to contain a batch of random numbers
        net.tops['value%d' % step].data[:, 0, 0, 0] = value
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
        # Concatenate the hidden output with the next input value 
        net.forward_layer(layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'value%d' % step]))
        # Run the LSTM for one more step
        net.forward_layer(layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['input_value', 'input_gate', 'forget_gate', 'output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler))

    # Add a fully connected layer with a bottom blob set to be the last used LSTM cell
    # Note that the network structure is now a function of the data
    net.forward_layer(layers.InnerProduct(name='ip',
        bottoms=['lstm%d_hidden' % (length - 1)],
        num_output=1, weight_filler=filler))
    # Add a label for the sum of the inputs
    net.forward_layer(layers.NumpyData(name='label',
        data=np.reshape(accum, (hyper['batch_size'], 1, 1, 1))))
    # Compute the Euclidean loss between the preiction and label, used for backprop
    loss = net.forward_layer(layers.EuclideanLoss(name='euclidean',
        bottoms=['ip', 'label']))
    return loss

def train(hyper):
    apollo.set_random_seed(hyper['random_seed'])
    if hyper['gpu'] is None:
        apollo.set_mode_cpu()
        logging.info('Using cpu device (pass --gpu X to train on the gpu)')
    else:
        apollo.set_mode_gpu()
        apollo.set_device(hyper['gpu'])
        logging.info('Using gpu device %d' % hyper['gpu'])
    apollo.set_logging_verbosity(hyper['loglevel'])

    net = apollo.Net()
    forward(net, hyper)
    network_path = '%s/network.jpg' % hyper['schematic_prefix']
    net.draw_to_file(network_path)
    logging.info('Drawing network to %s' % network_path)
    net.reset_forward()
    if 'weights' in hyper:
        logging.info('Loading weights from %s' % hyper['weights'])
        net.load(hyper['weights'])

    train_loss_hist = []
    for i in xrange(hyper['start_iter'], hyper['max_iter']):
        train_loss_hist.append(forward(net, hyper))
        net.backward()
        lr = (hyper['base_lr'] * hyper['gamma']**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper['clip_gradients'])
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

def evaluate_forward(net):
    length = 20
    net.forward_layer(layers.NumpyData(name='prev_hidden',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='prev_mem',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    filler = layers.Filler(type='uniform', min=-hyper['init_range'], max=hyper['init_range'])
    accum = np.array([0.])
    predictions = []
    for step in range(length):
        value = 0.5
        net.forward_layer(layers.NumpyData(name='value',
            data=np.array(value).reshape((1, 1, 1, 1))))
        accum += value
        prev_hidden = 'prev_hidden'
        prev_mem = 'prev_mem'
        net.forward_layer(layers.Concat(name='lstm_concat', bottoms=[prev_hidden, 'value']))
        net.forward_layer(layers.Lstm(name='lstm', bottoms=['lstm_concat', prev_mem],
            param_names=['input_value', 'input_gate', 'forget_gate', 'output_gate'],
            weight_filler=filler,
            tops=['next_hidden', 'next_mem'], num_cells=hyper['mem_cells']))
        net.forward_layer(layers.InnerProduct(name='ip', bottoms=['next_hidden'],
            num_output=1))
        predictions.append(float(net.tops['ip'].data.flatten()[0]))
        # set up for next prediction by copying LSTM outputs back to inputs
        net.tops['prev_hidden'].data_tensor.copy_from(net.tops['next_hidden'].data_tensor)
        net.tops['prev_mem'].data_tensor.copy_from(net.tops['next_mem'].data_tensor)
        net.reset_forward()
    return predictions

def eval(hyper):
    eval_net = apollo.Net()
    # evaluate the net once to set up structure before loading parameters
    evaluate_forward(eval_net)
    eval_net.load('%s/%d.h5' % (hyper['snapshot_prefix'], hyper['max_iter'] - 1))
    print evaluate_forward(eval_net)

def main():
    hyper = {}
    hyper['gpu'] = None
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['base_lr'] = 0.03
    hyper['momentum'] = 0.9
    hyper['clip_gradients'] = 0.1
    hyper['display_interval'] = 100
    hyper['max_iter'] = 5001
    hyper['snapshot_prefix'] = '/tmp'
    hyper['schematic_prefix'] = '/tmp'
    hyper['snapshot_interval'] = 1000
    hyper['random_seed'] = 21
    hyper['gamma'] = 0.5
    hyper['stepsize'] = 1000
    hyper['solver_mode'] = 'gpu'
    hyper['mem_cells'] = 1000
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = '/tmp'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--loglevel', default=3, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--weights', default=None, type=str)
    args = parser.parse_args()
    hyper.update({k:v for k, v in vars(args).iteritems() if v is not None})
    train(hyper)
    eval(hyper)

if __name__ == '__main__':
    main()
