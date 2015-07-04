#!/usr/bin/env python
import numpy as np
import random
import logging
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import apollo
from apollo import layers

def get_hyper():
    hyper = {}
    hyper['vocab_size'] = 10000
    hyper['batch_size'] = 1
    hyper['init_range'] = 0.1
    hyper['zero_symbol'] = hyper['vocab_size'] - 1
    hyper['unknown_symbol'] = hyper['vocab_size'] - 2
    hyper['test_interval'] = 100
    hyper['test_iter'] = 20
    hyper['base_lr'] = 2
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0.0
    hyper['clip_gradients'] = 0.24
    hyper['display_interval'] = 20
    hyper['max_iter'] = 20000
    hyper['snapshot_prefix'] = '/snapshots/lm'
    hyper['snapshot_interval'] = 1000
    hyper['random_seed'] = 22
    hyper['net'] = '/shared/u/apollo/examples/language_model/train_val.prototxt'
    hyper['gamma'] = 0.792
    hyper['stepsize'] = 2500
    hyper['solver_mode'] = 'gpu'
    hyper['mem_cells'] = 250
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = ''
    return hyper

hyper = get_hyper()

apollo.Caffe.set_random_seed(hyper['random_seed'])
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(0)
apollo.Caffe.set_logging_verbosity(3)

def get_data():
    data_source = '/shared/u/nlpcaffe/data/language_model/shuffled_train_indices.txt'
    while True:
        with open(data_source, 'r') as f:
            for x in f.readlines():
                yield x.strip().split(' ')

def pad_rnn(sentence_batch):
    max_len = max(len(x) for x in sentence_batch)
    result = []
    for x in sentence_batch:
        y = [int(z) if int(z) < hyper['unknown_symbol'] else hyper['unknown_symbol'] for z in x ]
        result.append(y + [hyper['zero_symbol']] * (max_len - len(x)))
    return result
    
def get_data_batch(data_iter):
    while True:
        raw_batch = [next(data_iter) for _ in range(hyper['batch_size'])]
        sentence_batch = np.array(pad_rnn(raw_batch))
        yield sentence_batch

counter = 0
def forward(net, sentence_batches):
    sentence_batch = next(sentence_batches)
    length = min(sentence_batch.shape[1], 10)

    filler = layers.Filler(type='constant', max=hyper['init_range'], min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='label',
        data=np.zeros((hyper['batch_size'] * length, 1, 1, 1))))
    hidden_concat_bottoms = []
    for step in range(length):
        net.forward_layer(layers.DummyData(name=('word%d' % step),
            shape=[hyper['batch_size'], 1, 1, 1]))
        word = sentence_batch[:, step]
        net.tops['word%d' % step].data[:,0,0,0] = word
        net.forward_layer(layers.Wordvec(name=('wordvec%d' % step), bottoms=['word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
        net.forward_layer(layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.forward_layer(layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate', 'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout%d' % step,
            bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16))
        hidden_concat_bottoms.append('dropout%d' % step)

    net.forward_layer(layers.Concat(name='hidden_concat',
        concat_dim=0, bottoms=hidden_concat_bottoms))
    net.tops['label'].data[:,0,0,0] = sentence_batch[:, :length].T.flatten()
    net.forward_layer(layers.InnerProduct(name='ip', bottoms=['hidden_concat'],
        num_output=hyper['vocab_size'], weight_filler=filler))
    loss = net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss',
        ignore_label=hyper['zero_symbol'], bottoms=['ip', 'label']))
    return loss

net = apollo.Net()

sentences = get_data()
sentence_batches = get_data_batch(sentences)

forward(net, sentence_batches)
net.reset_forward()
#net.load('/snapshots/lm_1000.h5')
train_loss_hist = []

for i in range(hyper['max_iter']):
    train_loss_hist.append(forward(net, sentence_batches))
    net.backward()
    lr = (hyper['base_lr'] / (hyper['gamma'])**(i // hyper['stepsize']))
    net.update(lr=lr, momentum=hyper['momentum'],
        clip_gradients=hyper['clip_gradients'], weight_decay=hyper['weight_decay'])
    if i % hyper['display_interval'] == 0:
        logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
    if i % hyper['test_interval'] == 0:
        #test_performance(net, test_net)
        pass
    if i % hyper['snapshot_interval'] == 0 and i > 0:
        filename = '%s_%d.h5' % (hyper['snapshot_prefix'], i)
        logging.info('Saving net to: %s' % filename)
        net.save(filename)
    if i % hyper['graph_interval'] == 0 and i > 0:
        sub = 100
        plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
        filename = '%strain_loss.jpg' % hyper['graph_prefix']
        logging.info('Saving figure to: %s' % filename)
        plt.savefig(filename)
