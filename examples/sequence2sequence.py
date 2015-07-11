"""
This example trains a sequence2sequence network to reverse the sentences found in the language model example.
"""
import numpy as np
import random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
apollo_root = os.environ['APOLLO_ROOT']

import simplejson as json
import apollo
import logging
from apollo import layers
from scipy.spatial.distance import cosine

def get_hyper():
    hyper = {}
    hyper['vocab_size'] = 10000
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['zero_symbol'] = hyper['vocab_size'] - 1
    hyper['unknown_symbol'] = hyper['vocab_size'] - 2
    hyper['test_interval'] = 100
    hyper['test_iter'] = 20
    hyper['base_lr'] = 1
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0.0
    hyper['clip_gradients'] = 0.24
    hyper['display_interval'] = 20
    hyper['max_iter'] = 2000000
    hyper['snapshot_prefix'] = '/tmp/lm'
    hyper['snapshot_interval'] = 10000
    hyper['random_seed'] = 22
    hyper['gamma'] = 0.792
    hyper['stepsize'] = 10000
    hyper['mem_cells'] = 250
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = ''
    hyper['max_len'] = 30
    return hyper

hyper = get_hyper()

apollo.Caffe.set_random_seed(hyper['random_seed'])
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(1)
apollo.Caffe.set_logging_verbosity(3)

def get_data():
    # You can download this file with bash ./data/language_model/get_lm.sh
    data_source = '%s/data/language_model/train_indices.txt' % apollo_root
    epoch = 0
    while True:
        with open(data_source, 'r') as f:
            for x in f.readlines():
                yield x.strip().split(' ')
        logging.info('epoch %s finished' % epoch)
        epoch += 1
    
def pad_batch(sentence_batch):
    max_len = min(max(len(x) for x in sentence_batch), hyper['max_len'])
    f_result = []
    b_result = []
    for x in sentence_batch:
        y = [int(z) if int(z) < hyper['unknown_symbol'] else hyper['unknown_symbol']
            for z in x[:max_len]]
        f_result.append(y + [hyper['zero_symbol']] * (max_len - len(x)))
        b_result.append(y[::-1] + [hyper['zero_symbol']] * (max_len - len(x)))
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
    length = target_batch.shape[1]

    filler = layers.Filler(type='uniform', max=hyper['init_range'],
            
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='source_lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    hidden_bottoms = ['source_lstm_seed']
    mem_bottoms = ['source_lstm_seed']
    hash_bottoms = []
    lengths = [min(len([1 for token in x if token != hyper['zero_symbol']]), hyper['max_len']) for x in source_batch]
    for step in range(length):
           
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
    
    hidden_concat_bottoms = []
    loss = []
    for step in range(length):
        if step == 0:
            prev_hidden = 'hidden_seed'
            prev_mem = 'mem_seed'
            prev_hidden = 'source_lstm%d_hidden' % (length - 1)
            prev_mem = 'source_lstm%d_mem' % (length - 1)
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

        hidden_concat_bottoms.append('dropout%d' % step)

        net.forward_layer(layers.NumpyData(name='label%d' % step,
            data=np.reshape(target_batch[:, step], (hyper['batch_size'], 1, 1, 1))))
        net.forward_layer(layers.InnerProduct(name='ip%d' % step, bottoms=['dropout%d' % step],
            param_names=['ip_weight', 'ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        loss.append(net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss%d' % step,
            ignore_label=hyper['zero_symbol'], bottoms=['ip%d' % step, 'label%d' % step])))
    return np.mean(loss)

net = apollo.Net()
test_net = apollo.Net()

sentences = get_data()
sentence_batches = get_data_batch(sentences)

forward(net, sentence_batches)
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

import pickle
import os
with open('%s/data/language_model/vocab.pkl' % os.environ['APOLLO_ROOT'], 'r') as f:
    vocab = pickle.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

#TODO: write eval_forward function. 
#eval_net = apollo.Net()
# eval_forward(eval_net)
#eval_net.load('%s_20000.h5' % hyper['snapshot_prefix'])
