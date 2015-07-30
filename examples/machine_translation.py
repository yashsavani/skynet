
"""
This example trains a sequence2sequence network to translate europarl Spanish-English.
It's not particularly good yet, but it does work and is a good reference for sequence2sequence.

Training set example:
Iteration 134300: 1.77240937874
input:   en la 22a sesion , celebrada el 25 de octubre , el secretario de la comision senalo a la atencion de los presentes una nota
output:  at the 25th meeting , on 19 october , the chairman , the representative the statement to a statement by the records see a EOS
target:  the 22nd meeting , on 25 october , the secretary of the committee drew attention to a note by the secretariat ( a / UNK
"""
import numpy as np
import random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os

import simplejson as json
import argparse
import apollo
import logging
from apollo import layers
apollo_root = '.'

def get_hyper():
    hyper = {}
    hyper['vocab_size'] = 10000 # use small vocab for example
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['pad_symbol'] = 2
    hyper['eos_symbol'] = 1
    hyper['unknown_symbol'] = 0
    hyper['test_interval'] = 100
    hyper['test_iter'] = 20
    hyper['base_lr'] = 1
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0.0
    hyper['clip_gradients'] = 0.24
    hyper['display_interval'] = 20
    hyper['max_iter'] = 2000000
    hyper['snapshot_prefix'] = '/tmp/mt'
    hyper['snapshot_interval'] = 10000
    hyper['random_seed'] = 22
    hyper['gamma'] = 0.792
    hyper['stepsize'] = 10000
    hyper['mem_cells'] = 250
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = ''
    hyper['max_len'] = 25
    return hyper

hyper = get_hyper()

apollo.Caffe.set_random_seed(hyper['random_seed'])
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(2)
apollo.Caffe.set_logging_verbosity(3)

data_prefix = '%s/data/machine_translation/build' % apollo_root

import pickle
import os
with open('%s/vocab.en.pkl' % data_prefix, 'r') as f:
    en_vocab = pickle.load(f)
with open('%s/ivocab.en.pkl' % data_prefix, 'r') as f:
    en_ivocab = pickle.load(f)
    en_ivocab[0] = 'UNK'
    en_ivocab[1] = 'EOS'
    en_ivocab[2] = 'PAD'
with open('%s/vocab.es.pkl' % data_prefix, 'r') as f:
    es_vocab = pickle.load(f)
with open('%s/ivocab.es.pkl' % data_prefix, 'r') as f:
    es_ivocab = pickle.load(f)
    es_ivocab[0] = 'UNK'
    es_ivocab[1] = 'EOS'
    es_ivocab[2] = 'PAD'

def get_data():
    # You can download this file with ...
    en_source = '%s/train.en.idx' % data_prefix
    es_source = '%s/train.es.idx' % data_prefix
    epoch = 0
    while True:
        with open(es_source, 'r') as f1:
            with open(en_source, 'r') as f2:
                for x, y in zip(f1.readlines(), f2.readlines()):
                    x_processed = map(int, x.strip().split(' '))
                    y_processed = map(int, y.strip().split(' '))
                    yield (x_processed, y_processed)
        logging.info('epoch %s finished' % epoch)
        epoch += 1

def padded_reverse(sentence):
    try:
        result = sentence[:sentence.index(hyper['pad_symbol'])][::-1]
    except:
        result = sentence[::-1]
    return result
    
def pad_batch(sentence_batch):
    source_len = min(max(len(x) for x, y in sentence_batch), hyper['max_len'])
    target_len = min(max(len(y) for x, y in sentence_batch), hyper['max_len'])
    f_result = []
    b_result = []
    for x, y in sentence_batch:
        x_clip = x[:source_len]
        y_clip = y[:target_len]
        f_result.append(x_clip + [hyper['pad_symbol']] * (source_len - len(x_clip)))
        b_result.append(y_clip[::-1] + [hyper['pad_symbol']] * (target_len - len(y_clip)))
    return np.array(f_result), np.array(b_result)
    
def get_data_batch(data_iter):
    while True:
        raw_batch = []
        for i in range(hyper['batch_size']):
            raw_batch.append(next(data_iter))
        sentence_batch = pad_batch(raw_batch)
        yield sentence_batch

def forward(net, sentence_batches):
    source_batch, target_batch = next(sentence_batches)

    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='source_lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    hidden_bottoms = ['source_lstm_seed']
    mem_bottoms = ['source_lstm_seed']
    lengths = [min(len([1 for token in x if token != hyper['pad_symbol']]), hyper['max_len']) for x in source_batch]
    for step in range(source_batch.shape[1]):
        net.forward_layer(layers.DummyData(name=('source_word%d' % step),
            shape=[hyper['batch_size'], 1, 1, 1]))
        if step == 0:
            prev_hidden = 'source_lstm_seed'
            prev_mem = 'source_lstm_seed'
        else:
            prev_hidden = 'source_lstm%d_hidden' % (step - 1)
            prev_mem = 'source_lstm%d_mem' % (step - 1)
        next_hidden = 'source_lstm%d_hidden' % (step)
        next_mem = 'source_lstm%d_mem' % (step)
        hidden_bottoms.append(next_hidden)
        mem_bottoms.append(next_mem)
        word = source_batch[:, step]
        net.tops['source_word%d' % step].data[:,0,0,0] = word
        wordvec = layers.Wordvec(name=('source_wordvec%d' % step),
            bottoms=['source_word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['source_wordvec_param'], weight_filler=filler)
        concat = layers.Concat(name='source_lstm_concat%d' % step,
            bottoms=[prev_hidden, 'source_wordvec%d' % step])
        lstm = layers.Lstm(name='source_lstm%d' % step,
            bottoms=['source_lstm_concat%d' % step, prev_mem],
            param_names=['source_lstm_input_value', 'source_lstm_input_gate',
                'source_lstm_forget_gate', 'source_lstm_output_gate'],
            tops=['source_lstm%d_hidden' % step, 'source_lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler)
        net.forward_layer(wordvec)
        net.forward_layer(concat)
        net.forward_layer(lstm)

    net.forward_layer(layers.CapSequence(name='hidden_seed', sequence_lengths=lengths,
        bottoms=hidden_bottoms))
    net.forward_layer(layers.CapSequence(name='mem_seed', sequence_lengths=lengths,
        bottoms=mem_bottoms))
    
    loss = []
    for step in range(target_batch.shape[1]):
        if step == 0:
            prev_hidden = 'hidden_seed'
            prev_mem = 'mem_seed'
            word = np.zeros(target_batch[:, 0].shape)
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
            word = target_batch[:, step - 1]
        word = layers.NumpyData(name=('word%d' % step),
            data=np.reshape(word, (hyper['batch_size'], 1, 1, 1)))
        wordvec = layers.Wordvec(name=('wordvec%d' % step),
            bottoms=['word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['source_wordvec_param'], weight_filler=filler)
        concat = layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'wordvec%d' % step])
        lstm = layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler)
        dropout = layers.Dropout(name='dropout%d' % step,
            bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16)
        
        net.forward_layer(word)        
        net.forward_layer(wordvec)   
        net.forward_layer(concat)
        net.forward_layer(lstm)
        net.forward_layer(dropout)

        net.forward_layer(layers.NumpyData(name='label%d' % step,
            data=np.reshape(target_batch[:, step], (hyper['batch_size'], 1, 1, 1))))
        net.forward_layer(layers.InnerProduct(name='ip%d' % step, bottoms=['dropout%d' % step],
            param_names=['ip_weight', 'ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        loss.append(net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss%d' % step,
            ignore_label=hyper['pad_symbol'], bottoms=['ip%d' % step, 'label%d' % step])))
        loss.append(net.forward_layer(layers.Softmax(name='softmax%d' % step,
            ignore_label=hyper['pad_symbol'], bottoms=['ip%d' % step])))
    return np.mean(loss)

def train():
    net = apollo.Net()
    test_net = apollo.Net()

    sentences = get_data()
    sentence_batches = get_data_batch(sentences)

    forward(net, sentence_batches)
    if hyper['weights'] is not None:
        net.load(hyper['weights'])
    net.reset_forward()
    train_loss_hist = []

    for i in range(hyper['max_iter']):
        train_loss_hist.append(forward(net, sentence_batches))
        net.backward()
        lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper['clip_gradients'], weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
            output = []
            target = []
            source = []
            for step in range(hyper['max_len']):
                try:
                    output.append(np.argmax(net.tops['softmax%d' % step].data[0].flatten()))
                    target.append(np.int(net.tops['word%d' % step].data[0].flatten()))
                    source.append(int(net.tops['source_word%d' % step].data[0].flatten()[0]))
                except:
                    break
            logging.info('input:\t' + ' '.join(es_ivocab[x] for x in source))
            logging.info('output:\t' + ' '.join(en_ivocab[x] for x in padded_reverse(output)))
            logging.info('target:\t' + ' '.join(en_ivocab[x] for x in padded_reverse(target)))
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
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    hyper['weights'] = args.weights
    train()

main()
#TODO: write eval_forward function and live translation option. 
