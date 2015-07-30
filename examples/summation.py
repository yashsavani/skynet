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

def eval():
    eval_net = apollo.Net()
    # evaluate the net once to set up structure before loading parameters
    evaluate_forward(eval_net)
    eval_net.load('%s_%d.h5' % (hyper['snapshot_prefix'], hyper['max_iter'] - 1))
    print evaluate_forward(eval_net)

def main():
    hyper = apollo.default_hyper(momentum=0.9,
        clip_gradients=0.1,
        display_interval=100,
        max_iter=5001,
        snapshot_interval=1000,
        random_seed=7,
        gamma=0.5,
        stepsize=1000,
        graph_interval=1000)
    hyper['base_lr'] = 0.03
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['mem_cells'] = 1000
    apollo.update_hyper(hyper, apollo.default_parser().parse_args())
    apollo.train(hyper, forward=forward)
    eval()

if __name__ == '__main__':
    main()
