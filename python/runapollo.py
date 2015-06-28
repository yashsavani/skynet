#!/usr/bin/env python
import apollo
import argparse
from apollo.layers import ConvLayer, DropoutLayer, DummyDataLayer, NumpyDataLayer, SoftmaxLayer, SoftmaxWithLossLayer, LstmLayer, ConcatLayer, InnerProductLayer, WordvecLayer
import config
import h5py
import numpy as np
import random

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
        y = [int(z) if int(z) < config.unknown_symbol else config.unknown_symbol for z in x ]
        result.append(y + [config.zero_symbol] * (max_len - len(x)))
    return result
    
def get_data_batch(data_iter):
    while True:
        raw_batch = [next(data_iter) for _ in range(config.batch_size)]
        sentence_batch = np.array(pad_rnn(raw_batch))
        yield sentence_batch

counter = 0
def iter(net, sentence_batches):
    sentence_batch = next(sentence_batches)
    #print sentence_batch
    mem_cells = 500
    length = min(sentence_batch.shape[1], 50)
    #print sentence_batch
    #print length
    init_range = 0.1

    net.forward_layer(NumpyDataLayer(name='lstm_seed', data=np.zeros((config.batch_size, mem_cells, 1, 1))))
    #net.forward_layer(DummyDataLayer(name='label', tops=['label'], shape=[config.batch_size * length, 1, 1, 1]))
    net.forward_layer(NumpyDataLayer(name='label', data=np.zeros((config.batch_size * length, 1, 1, 1))))
    #print net.blobs['label'].data().shape
    hidden_concat_bottoms = []
    for step in range(length):
        net.forward_layer(DummyDataLayer(name=('word%d' % step), shape=[config.batch_size, 1, 1, 1]))
        word = sentence_batch[:, step]
        net.blobs['word%d' % step].data()[:,0,0,0] = word
        net.forward_layer(WordvecLayer(name=('wordvec%d' % step), bottoms=['word%d' % step], dimension=mem_cells, vocab_size=config.vocab_size, init_range=init_range))
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
        net.forward_layer(ConcatLayer(name='lstm_concat%d' % step, bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.forward_layer(LstmLayer(name='lstm%d' % step, bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate', 'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step], num_cells=mem_cells, init_range=init_range))
        net.forward_layer(DropoutLayer(name='dropout%d' % step, bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16))
        hidden_concat_bottoms.append('dropout%d' % step)

    net.forward_layer(ConcatLayer(name='hidden_concat', concat_dim=0, bottoms=hidden_concat_bottoms))
    net.blobs['label'].data()[:,0,0,0] = sentence_batch[:, :length].T.flatten()
    net.forward_layer(InnerProductLayer(name='ip', bottoms=['hidden_concat'], num_output=config.vocab_size, init_range=init_range))
    #print 'ip_shape: ', net.blobs['ip'].shape()
    #net.forward_layer(SoftmaxLayer(name='softmax', bottoms=['ip']))
    loss = net.forward_layer(SoftmaxWithLossLayer(name='softmax_loss', ignore_label=config.zero_symbol, bottoms=['ip', 'label']))
    #net.forward_layer(EuclideanLossLayer(name='euclidean', bottoms=['ip', 'label']))
    net.backward()
    net.update(lr=20, momentum=0.0, clip_gradients=.24)
    #print net.blobs['softmax'].data()[0]
    #print loss
    return loss #* config.batch_size * length / sum(1 for x in sentence_batch.flatten() if x != config.zero_symbol)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default=0)
    args = parser.parse_args()
    apollo.Caffe.set_logging_verbosity(int(args.loglevel)) # turn off logging
    sentences = get_data()
    sentence_batches = get_data_batch(sentences)
    net = apollo.Net()
    net.set_phase_train()
    apollo.Caffe.set_random_seed(10)
    apollo.Caffe.set_mode_gpu()
    apollo.Caffe.set_device(0)

    #import time
    #shape = [3,256,256]
    #numpy_layer = NumpyDataLayer(name='numpy', tops=['numpy'], data=np.reshape(np.arange(shape[0] * shape[1] * shape[2]), shape))
    #net.forward_layer(numpy_layer)
    #net.backward()
    #net.update(lr=20)
    #print net.blobs['numpy'].data().shape
    #shape = [2,256,256]
    #numpy_layer = NumpyDataLayer(name='numpy', tops=['numpy'], data=np.reshape(np.arange(shape[0] * shape[1] * shape[2]), shape))
    #net.forward_layer(numpy_layer)
    #net.backward()
    #net.update(lr=20)
    #print net.blobs['numpy'].data().shape
    #return
    #print time.time() - start
    #print net.blobs['numpy'].data()

    net.reshape_only = True
    iter(net, sentence_batches)
    net.reshape_only = False
    error = 0
    display_interval = 10
    for i in range(100000):
        loss = iter(net, sentence_batches)
        error += loss / display_interval
        if i % display_interval == 0 and i > 0:
            print 'Iteration %d: %s' % (i, np.mean(error))
            error = 0

    #net.save('/tmp/a.h5')

if __name__ == '__main__':
    main()
