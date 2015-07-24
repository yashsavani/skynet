
# coding: utf-8
import numpy as np
import random
import matplotlib
import os
import simplejson as json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
apollo_root = os.environ['APOLLO_ROOT']

import apollo
import logging
from apollo import layers

import pickle
import os

def get_hyper():
    hyper = {}
    hyper['vocab_size'] = 256
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['zero_symbol'] = hyper['vocab_size'] - 1
    hyper['unknown_symbol'] = hyper['vocab_size'] - 2
    hyper['test_interval'] = 100
    hyper['test_iter'] = 20
    hyper['base_lr'] = 0.2
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0.0
    hyper['clip_gradients'] = 20
    hyper['display_interval'] = 100
    hyper['max_iter'] = 10000
    hyper['snapshot_prefix'] = '/tmp/char'
    hyper['snapshot_interval'] = 1000
    hyper['random_seed'] = 22
    hyper['gamma'] = 0.8
    hyper['graph_interval'] = 1000
    hyper['stepsize'] = 2500
    hyper['mem_cells'] = 1000
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = ''
    hyper['i_temperature'] = 1.5
    return hyper

hyper = get_hyper()

apollo.Caffe.set_random_seed(hyper['random_seed'])
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(0)
apollo.Caffe.set_logging_verbosity(3)

with open('%s/data/language_model/vocab.pkl' % apollo_root, 'r') as f:
    vocab = pickle.load(f)
ivocab = {v: k for k, v in vocab.items()}

def get_data():
    data_source = '%s/data/char_model/reddit_ml.txt' % apollo_root
    if not os.path.exists(data_source):
        raise IOError('You must download the data with ./data/character_model/get_reddit_lm.sh')
    epoch = 0
    while True:
        with open(data_source, 'r') as f:
            for x in f.readlines():
                data = json.loads(x)
                if len(data['body']) == 0:
                    continue
                yield data
        logging.info('epoch %s finished' % epoch)
        epoch += 1

def get_data_batch(data_iter):
    while True:
        batch = []
        for i in range(hyper['batch_size']):
            batch.append(next(data_iter))
        yield batch

def pad_batch(sentence_batch):
    max_len = max(len(x) for x in sentence_batch)
    result = []
    for sentence in sentence_batch:
        chars = [min(ord(c), 255) for c in sentence] 
        result.append(chars + [hyper['zero_symbol']] * (max_len - len(sentence)))
    return result

def forward(net, sentence_batches):
    batch = next(sentence_batches)
    sentence_batch = np.array(pad_batch([x['body'] for x in batch]))
    length = min(sentence_batch.shape[1], 100)
    assert length > 0

    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='label',
        data=np.zeros((hyper['batch_size'] * length, 1, 1, 1))))
    loss = []
    for step in range(length):
        net.forward_layer(layers.DummyData(name=('word%d' % step),
            shape=[hyper['batch_size'], 1, 1, 1]))
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
            word = np.zeros(sentence_batch[:, 0].shape)
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
            word = sentence_batch[:, step - 1]
        net.tops['word%d' % step].data[:,0,0,0] = word
        net.forward_layer(layers.Wordvec(name=('wordvec%d' % step),
            bottoms=['word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.forward_layer(layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout%d' % step,
            bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16))
        
        label = np.reshape(sentence_batch[:, step], (hyper['batch_size'], 1, 1, 1))
        net.forward_layer(layers.NumpyData(name='label%d' % step,
            data=label))
        net.forward_layer(layers.InnerProduct(name='ip%d' % step, bottoms=['dropout%d' % step],
            param_names=['softmax_ip_weights', 'softmax_ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        loss.append(net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss%d' % step,
            ignore_label=hyper['zero_symbol'], bottoms=['ip%d' % step, 'label%d' % step])))

    return np.mean(loss)

def eval_performance(net):
    eval_net = apollo.Net()
    eval_forward(eval_net)
    eval_net.copy_params_from(net)
    output_words = eval_forward(eval_net)
    print ''.join([chr(x) for x in output_words])

def softmax_choice(data):
    try:
        return np.random.choice(range(len(data.flatten())), p=data.flatten())
    except:
        return np.argmax(data.flatten())

def eval_forward(net):
    output_words = []
    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_hidden_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='lstm_mem_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    length = 150
    for step in range(length):
        net.forward_layer(layers.NumpyData(name=('word'),
            data=np.zeros((1, 1, 1, 1))))
        prev_hidden = 'lstm_hidden_prev'
        prev_mem = 'lstm_mem_prev'
        word = np.zeros((1, 1, 1, 1))
        if step == 0:
            output = ord('.')
        else:
            output = softmax_choice(net.tops['softmax'].data)
        output_words.append(output)
        net.tops['word'].data[0,0,0,0] = output
        net.forward_layer(layers.Wordvec(name=('wordvec'),
            bottoms=['word'],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat',
            bottoms=[prev_hidden, 'wordvec']))
        net.forward_layer(layers.Lstm(name='lstm',
            bottoms=['lstm_concat', prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm_hidden_next', 'lstm_mem_next'],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout',
            bottoms=['lstm_hidden_next'], dropout_ratio=0.16))

        net.forward_layer(layers.InnerProduct(name='ip', bottoms=['dropout'],
            param_names=['softmax_ip_weights', 'softmax_ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        net.tops['ip'].data[:] *= hyper['i_temperature']
        net.forward_layer(layers.Softmax(name='softmax',
            ignore_label=hyper['zero_symbol'], bottoms=['ip']))
        net.tops['lstm_hidden_prev'].data_tensor.copy_from(net.tops['lstm_hidden_next'].data_tensor)
        net.tops['lstm_mem_prev'].data_tensor.copy_from(net.tops['lstm_mem_next'].data_tensor)
        net.reset_forward()
    return output_words

net = apollo.Net()

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
    if i % hyper['test_interval'] == 0:
        eval_performance(net)
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

'''
Sample output:
================================================================================

2015-07-17 18:07:20,342 - INFO - Iteration 0: 5.53014346123
.
   (t(b*r¬p:N|O2&֡ fӚµø ]WÁD­ yo n  sr      i                g       e       u       i  i  s     w
2015-07-17 18:09:12,321 - INFO - Iteration 100: 4.32106374675
. hot t dt cit ti t t os es tht po dat to ths to t we t t pas to t os wop to is t as al d t is t l if to tos ad toe d t to  wet do to mabct to u s t m
2015-07-17 18:11:06,470 - INFO - Iteration 200: 2.5238280553
. I on sep learne deff is   I weand here dancle &gt; I h0s mearnele ary comperta or an Mge and heruld sige is upe and a from aptral" In and if learne
2015-07-17 18:13:01,012 - INFO - Iteration 300: 2.05137892483
. Is of the to the take the to ne of the with the tell to to the extell used to hat would be to to lobate at to the ating to ge the to the to be fun t
2015-07-17 18:14:55,659 - INFO - Iteration 400: 1.81043039544
. Rets on rades an eddit librarient and wand on sparifican in image a rangle with eir amain an in experian in pata an in Can from it of in way of line
2015-07-17 18:16:49,375 - INFO - Iteration 500: 1.69140332802
. I think they don't too with they they deep the faid to the implementation the top top a lot to expect the paper these but the fant the series the te
2015-07-17 18:18:40,795 - INFO - Iteration 600: 1.61146559111
. I then the ML convolutionally bencope for your defined complettensure is all the finest to see a second comes trated and convolution to see the bool
2015-07-17 18:20:35,407 - INFO - Iteration 700: 1.563197357
. It was have one sparse of all it sparsity.

&gt;Implement it is simple metric and stration, for your good contries on the OS is a lot of the startio
2015-07-17 18:22:30,104 - INFO - Iteration 800: 1.51405745667
. The decading statistics is too compart is it some post is very matrix optimize and the optimize approaches are the planation of the with to optimize
2015-07-17 18:24:25,085 - INFO - Iteration 900: 1.4869060873
. It was all that it would be able to as specifically aboutdit about "uting a sely what you are all. If you are using the mine. I don't understand it
2015-07-17 18:26:19,949 - INFO - Iteration 1000: 1.46797525926
. I have a m Did network the automatic in a bit better network mark but it will looking for a description of this case but he's deviawed a pretty much
2015-07-17 18:26:21,390 - INFO - Saving net to: /tmp/char_1000.h5
2015-07-17 18:26:21,420 - INFO - Saving figure to: train_loss.jpg
2015-07-17 18:28:12,703 - INFO - Iteration 1100: 1.42537454589
. You can imagine the corport the subject these thing that me things on eavie does the same thing in the sense to the point is pretty generalized the
2015-07-17 18:30:07,193 - INFO - Iteration 1200: 1.41676740446
. I was looking for from what is pretty world get a lot of researcher and I do not look at matrix for probability is program, on my advers for Machine
2015-07-17 18:32:01,561 - INFO - Iteration 1300: 1.4066332077
. I try a few impressive by "college mathabout in the fold of the Book at I think you are artificial dataset) and the features of far as a problem of
2015-07-17 18:33:54,892 - INFO - Iteration 1400: 1.38168845002
. The point of results to make a sense, or are uninteresting to the probability of NB and I am the subreddit.  What is it. The author is not to get a
2015-07-17 18:35:50,934 - INFO - Iteration 1500: 1.37484624689
. I may be able to probably don't expace the first techniques at the fact that are over what the problems at a large architooks are the best implement
2015-07-17 18:37:43,593 - INFO - Iteration 1600: 1.36943356241
. If you want to assume that it, and you cannot include many. Also a minimization of anyone have any such a prebuits of what I can include from when y
2015-07-17 18:39:39,116 - INFO - Iteration 1700: 1.3477439496
. In the feedfores then this is why I have a more well with the encertan one of the courses. It is in the entire concepts of the these fine more prett
2015-07-17 18:40:17,331 - INFO - epoch 0 finished
2015-07-17 18:41:33,742 - INFO - Iteration 1800: 1.34326361285
. So that was no take sure straightforward published data reforts on the same thing that have a label on the Gaussian companies are near and a little
2015-07-17 18:43:28,706 - INFO - Iteration 1900: 1.33609817184
.  They're just defined the solution of each data sets or its 100.0, you. Dear which has been a features and is that you're being able to do this so I
2015-07-17 18:45:22,925 - INFO - Iteration 2000: 1.32598233822
. I prove a very line for a gradient for a gradient detection is a great thread for contance when the method when you should see her that to the most
2015-07-17 18:45:24,526 - INFO - Saving net to: /tmp/char_2000.h5
2015-07-17 18:45:24,549 - INFO - Saving figure to: train_loss.jpg
2015-07-17 18:47:17,789 - INFO - Iteration 2100: 1.31345041214
. It seems that his model (   - the concept on the next deep learning that the large data is an enough concept in a number of data in ML: http://www.r
2015-07-17 18:49:10,947 - INFO - Iteration 2200: 1.30505523615
.  I only adjust the connect to check it stitutes on the unit of the whoch is the interest is the formulation with special project to do the supervise
2015-07-17 18:51:04,762 - INFO - Iteration 2300: 1.30047465323
. What is extreme on the studing, in it, if you help to be the distribution. Machine learning for making my material just something like it better to
2015-07-17 18:53:00,581 - INFO - Iteration 2400: 1.28053402412
.  I expected the data from many classifier.  Does more often and that paper has a specialization that did you need

That reason concepts are complete
2015-07-17 18:54:54,889 - INFO - Iteration 2500: 1.28011639886
.  It seems like he can be answered to see some of the field:

http://www.csal.mit.edu/~bskerne/n/machine-learning------------------------------------
2015-07-17 18:56:50,186 - INFO - Iteration 2600: 1.24369505327
. How are that would be to get sure if there's papers that is pretty looking for that.  It probably think about the post to be abmutable in a competit
2015-07-17 18:58:43,809 - INFO - Iteration 2700: 1.25127968567
. Consider a large difference would be the challenge in the most neuron here

http://github.com/machinelearning/) if you have different the? I would l
2015-07-17 19:00:38,859 - INFO - Iteration 2800: 1.2350651291
. I think there used the sense of the paper control them allocated sequences every info on books about the sample is used to see anything entire senti
2015-07-17 19:02:34,411 - INFO - Iteration 2900: 1.23320884169
. I'm not that they want to mean it's random forests (with a random finster. :) If you are trying to write a small?

I have to be dear it's because it
2015-07-17 19:04:29,484 - INFO - Iteration 3000: 1.2382998897
.

the competitions may represent an introduction to do you are the same reduce to deep learning response is like the function, in math behind when th
2015-07-17 19:04:31,031 - INFO - Saving net to: /tmp/char_3000.h5
2015-07-17 19:04:31,063 - INFO - Saving figure to: train_loss.jpg
2015-07-17 19:06:23,029 - INFO - Iteration 3100: 1.22825260414
...statistical machine learning algorithm is that there is correlated using a size parameters in the centroid this on the talk of statistical machine
2015-07-17 19:08:16,278 - INFO - Iteration 3200: 1.23191634175
.  I didn't mean by check in statistical Boyds to a machine learning algorithms already is interest on this assignment on a problem if you could decid
2015-07-17 19:10:10,599 - INFO - Iteration 3300: 1.22358880083
. I wonder what he was going to be confident for the time run this on the same selection. It is not confused for doing this set?  The question is that
2015-07-17 19:12:04,967 - INFO - Iteration 3400: 1.21777939103
. If you are competed to multiple times. The weights, doesn't have a big about the problem is that it's about training.

The goal of the goal is the
2015-07-17 19:13:21,408 - INFO - epoch 1 finished
2015-07-17 19:14:00,152 - INFO - Iteration 3500: 1.22432422543
.  You could try this service. You can find a catching Lawout is orisal, however, and time skill with good links are a big for computational and comme
2015-07-17 19:15:55,905 - INFO - Iteration 3600: 1.21127504714
. I am using the post and no theory, you are in a bit of the simple model for meaning things but it was an un first subreddit but I'm going to complet
2015-07-17 19:17:47,369 - INFO - Iteration 3700: 1.216438207
. I am not sure if they the slides of this version of sigmoid to the course on the speed  one then can improve mathematics on the tech recent as the o
2015-07-17 19:19:42,154 - INFO - Iteration 3800: 1.2176422332
. If the way they step by doing the top 10 hours probably not at Google learned as an answer to state-of-to-a-max-video-from-elaction/tour/site/0 file
2015-07-17 19:21:36,671 - INFO - Iteration 3900: 1.20079063803
. But I don't know what it is been detecting the stort stuff that are pretty competing to show the same training code in the classifier on the enterin
2015-07-17 19:23:30,493 - INFO - Iteration 4000: 1.20368343532
.

I bet it worked through Alex Optimization and Machine Learning is in a whole brain to do a couple of books in the serves a big decomposition of the
2015-07-17 19:23:31,936 - INFO - Saving net to: /tmp/char_4000.h5
2015-07-17 19:23:31,964 - INFO - Saving figure to: train_loss.jpg
2015-07-17 19:25:25,061 - INFO - Iteration 4100: 1.18212791257
.

Even what giving the interface control out thermanis of machine learning. The abstract would need to get the content of time size!  I expected as i
2015-07-17 19:27:18,876 - INFO - Iteration 4200: 1.19622194338
.

The also you were right. We should get a model was tricky with it in the distribution of algorithms are along the contents as a few classifiers, th
2015-07-17 19:29:11,174 - INFO - Iteration 4300: 1.17974757243
. If bio do a PhD student with a data science and that didn't work with sum-of-providing the sites at all when I want to mind a couple. I see complexi
2015-07-17 19:31:04,813 - INFO - Iteration 4400: 1.19553985373
. I've been making the past of the site or only to do and then reduce our words or pretty much some classification to this far?org. This is somewhat m
2015-07-17 19:32:59,907 - INFO - Iteration 4500: 1.19152703595
.  It's a follow up track, but you want to read the theory.  I use a linear machine learning that were disallow for the feature engineering while it's
2015-07-17 19:34:54,868 - INFO - Iteration 4600: 1.1874997512
.

I think there are some points for Python, the answer is not a lot on the author of this site.  I think you want to have many problems with a simila
2015-07-17 19:36:47,692 - INFO - Iteration 4700: 1.18594685474
. It should be more about jobs on this properties in a thing to be interesting to this course. I don't know more about Statistics Rachin-Algorithms, a
2015-07-17 19:38:40,416 - INFO - Iteration 4800: 1.18281046387
. This talks/ consider to develop a amount of concepts that happen to account much and statistics and I don't have a lot of supervised learning with s
2015-07-17 19:40:34,721 - INFO - Iteration 4900: 1.18368957267
. It could be interesting to see if I would suggest each other people used to generate something like the model you come about manager to the imagenet
2015-07-17 19:42:30,907 - INFO - Iteration 5000: 1.18449374014
. I'm specialized for what i am totally a little  of a very other parameters for this up in Theano for the ML! It's a bit the caret with the things I
2015-07-17 19:42:32,322 - INFO - Saving net to: /tmp/char_5000.h5
2015-07-17 19:42:32,360 - INFO - Saving figure to: train_loss.jpg
2015-07-17 19:44:25,226 - INFO - Iteration 5100: 1.16339751153
.

1. It's a very garging and problem. Algorithms wrote the data between different articles to me. All the post is best to investing the decision-bu
2015-07-17 19:46:21,129 - INFO - Iteration 5200: 1.15857841768
. I went to hirtographic videos and conditional linear algebra" (computers and see which internships wouldn't very very very much better.

One is pret
2015-07-17 19:46:23,378 - INFO - epoch 2 finished
2015-07-17 19:48:13,249 - INFO - Iteration 5300: 1.15560188379
. I could start with the ideas or linear algebra is good. I'm assuming that the description of the neural network set is at least the 'said that there
2015-07-17 19:50:06,657 - INFO - Iteration 5400: 1.15969462777
. Any thesis is a good idea. I was thinking of reading about this context from some of the Many benchmarks are already been a criticizing term I've kn
2015-07-17 19:52:01,052 - INFO - Iteration 5500: 1.15685295645
...I've been up at a dataset in research in your own answer. Sometimes almost typically are saying that we ashared it to see a more clear to a simple
2015-07-17 19:53:55,722 - INFO - Iteration 5600: 1.14585975433
. I'm unfortunately sure this is pretty sure it was interested in Jueations, this is that initialisation in my computer experts well as the weights to
2015-07-17 19:55:50,532 - INFO - Iteration 5700: 1.14764828905
... it's a good code for the link. I am probably going to say that it explains a learning confusion or the course on CMU. I think that the first testi
2015-07-17 19:57:46,112 - INFO - Iteration 5800: 1.13869841329
... (and there_techniques the shite. There is a [Cross-validation technique](http://en.wikipedia.org/wiki/Index_harration).

http://www.cs.workington.
2015-07-17 19:59:39,734 - INFO - Iteration 5900: 1.13848282306
.  I don't know of any social problems so I can see what we think is integrated to a bit a faster students for the data (as a function of the data, an
2015-07-17 20:01:34,261 - INFO - Iteration 6000: 1.13112327975
. And if you shouldn't seem to be saying that it is about the best one calculation that it has a biology page.  This is good back to following theory,
2015-07-17 20:01:35,742 - INFO - Saving net to: /tmp/char_6000.h5
2015-07-17 20:01:35,767 - INFO - Saving figure to: train_loss.jpg
2015-07-17 20:03:29,538 - INFO - Iteration 6100: 1.13798405777
. People were watching this on the text on a consequence and even using SVMs and learning the text regression (I think the current compare with ML. I'
2015-07-17 20:05:25,437 - INFO - Iteration 6200: 1.13909799457
.  If you don't have any way to collect analysis is a similar project as an undergrad students done in the first time in the field of requirements tha
2015-07-17 20:07:20,039 - INFO - Iteration 6300: 1.12460453946
. There are some noise and doing a simple and then attann, no complicate the value of it was to add this?

If you don't see more information that is a
2015-07-17 20:09:12,275 - INFO - Iteration 6400: 1.12964016591
. That's non-based on the standard information by going on this with the stot of machine learning again in C++ so na layer?  You can work with a few i
2015-07-17 20:11:07,530 - INFO - Iteration 6500: 1.13722489769
. It's talking about this to the book, but we wanted to compare that the effect and machine learning courses has to develop some step to be made stude
2015-07-17 20:13:01,991 - INFO - Iteration 6600: 1.12890128931
. At the source for me and I considered the absolution type in the video lectures and see if the performance is that the great area of machine learnin
2015-07-17 20:14:57,459 - INFO - Iteration 6700: 1.12928884896
. I'm curious why this is very good.  Also, they mean best better with classification and windows in normalizing problems in a neural network assessin
2015-07-17 20:16:51,631 - INFO - Iteration 6800: 1.1348254465
. I wouldn't say that the top generation involution in general applications are some things that one thing is through the state-of-the-art application
2015-07-17 20:18:44,848 - INFO - Iteration 6900: 1.12577429968
. For example, the contains with a deep learning detail, and the results would be low-rank-- perform a choice of data in terms of relative to the scal
2015-07-17 20:19:25,214 - INFO - epoch 3 finished
2015-07-17 20:20:40,818 - INFO - Iteration 7000: 1.13307629534
...  this is the cost standard linear regression of their features.  I'm not following a lot of statistics or algorithms. In something from that probl
2015-07-17 20:20:41,952 - INFO - Saving net to: /tmp/char_7000.h5
2015-07-17 20:20:41,979 - INFO - Saving figure to: train_loss.jpg
2015-07-17 20:22:36,923 - INFO - Iteration 7100: 1.13320605733
. It is a bit of a phone as its options. I also want a few days ago. I have not actually understood how to put it to the probability.

Also the extens
2015-07-17 20:24:31,364 - INFO - Iteration 7200: 1.12568839595
. Internship... I'm glad to be fairly dense. It looks like it would be interesting to me. But I can ask who know that company here talks about using m
2015-07-17 20:26:25,249 - INFO - Iteration 7300: 1.12627870291
.  It seems that you posted that you are right that I have been playing along with it by the training set. You want to do a whitening! :)

http://mlr.
2015-07-17 20:28:19,565 - INFO - Iteration 7400: 1.12330577616
. This is a random way to become a predictive recommended semester as the input feature extraction reviews. I think it's a great comment.  Although I
2015-07-17 20:30:15,316 - INFO - Iteration 7500: 1.11910799143
. If you find a suggestion is possible that is a good tutorial on a systems (that is a project part of I can't comment on a good point :) :) It is onl
2015-07-17 20:32:10,906 - INFO - Iteration 7600: 1.10255500405
. It's not in the same thing. I am also reading this. Maybe this is really interesting. The field of computing separation works for converting to me t
2015-07-17 20:34:04,730 - INFO - Iteration 7700: 1.10075897489
... general systems for all the trained analytics states. I have tried it to deep learning methods, sequences and you would be against the function of
2015-07-17 20:35:59,942 - INFO - Iteration 7800: 1.09803501746
.  Good luck.

Also I see what I'm trying to learn and determine is that it's nice if there are several visualizations, the similarity descriptions ar
2015-07-17 20:37:54,387 - INFO - Iteration 7900: 1.10355751079
. I was just wanted to see how these courses is a with probabilities at the end of the college standardization would be different from the oher it is
2015-07-17 20:39:48,497 - INFO - Iteration 8000: 1.09162933975
. Thanks for the input. Do you have a shell that did they want to start researching backwards a lot about the name. I have edited the tester and not t
2015-07-17 20:39:50,098 - INFO - Saving net to: /tmp/char_8000.h5
2015-07-17 20:39:50,129 - INFO - Saving figure to: train_loss.jpg
2015-07-17 20:41:44,657 - INFO - Iteration 8100: 1.09220773654
. You may want to comph, it was a lot of parameters of a contest of a computational statistical machine learning researchers in the article is the nex
2015-07-17 20:43:40,167 - INFO - Iteration 8200: 1.0972963648
. The author of the site have to take the 3 training and model, and the learning rate between the same goes reported between the Studies. The quality
2015-07-17 20:45:34,692 - INFO - Iteration 8300: 1.09426901967
. It looks like it is a factor like machine learning researchers to say isn't patent to me with R machines compared to determining the best scenarios
2015-07-17 20:47:30,066 - INFO - Iteration 8400: 1.09516951873
. I read [this supervised Learning Networks](http://www.cs.cmu.edu/~tom/man/statistical-neural-network-information-details.html) gets a bit the trick
2015-07-17 20:49:22,141 - INFO - Iteration 8500: 1.09398647348
. It doesn't have a lot of ideas, but it seems to be a pretty good test think you might want to try and see what I'm looking for. I like this evolutio
2015-07-17 20:51:15,491 - INFO - Iteration 8600: 1.08885952148
. In this difference. It sounds like the input is going to be soon (out of the `fixed size of this in the parameters and Attransly) is the first thing
2015-07-17 20:52:35,065 - INFO - epoch 4 finished
2015-07-17 20:53:10,289 - INFO - Iteration 8700: 1.09200250103
. The reason would be easier to incorporate it as a science for end-date then it takes a linear classifier on the algorithm for the stream in other pa
2015-07-17 20:55:06,282 - INFO - Iteration 8800: 1.08475644654
. It's more than just a simple decision tree, I think about non parameterization. The author of the outputs of its are precisely. I see How many answe
2015-07-17 20:57:02,300 - INFO - Iteration 8900: 1.09230111222
. It seems a little confused by giving a control over time to have a language for a subsets of deep learning. Is there any way to apply machine learni
2015-07-17 20:58:56,091 - INFO - Iteration 9000: 1.09399662586
. If you even remove it to learning the net also feed forward to the states and all the last patently. I'm taking to know a bit of more than the code,
2015-07-17 20:58:57,609 - INFO - Saving net to: /tmp/char_9000.h5
2015-07-17 20:58:57,633 - INFO - Saving figure to: train_loss.jpg
2015-07-17 21:00:50,947 - INFO - Iteration 9100: 1.08128080166
. I went through the next link to the hiling method with analytics on up to it and it was pretty straightforward to help, but the interfaces aren't a
2015-07-17 21:02:47,020 - INFO - Iteration 9200: 1.08063782365
.  It shows that there is an interview in the end you could just take a look at the courses on Reddit. I don't understand your question.  You should r
2015-07-17 21:04:41,484 - INFO - Iteration 9300: 1.07384912248
. I have a good degree with some good datasets you are looking for a biggest post of how there would make me suggest that constructing a lot of data w
2015-07-17 21:06:35,560 - INFO - Iteration 9400: 1.087753241
. Also, if you're interested in this fortunate research. Good to know what you're doing. If you want to do a lot of functions and basic connection sys
2015-07-17 21:08:26,743 - INFO - Iteration 9500: 1.0719556538
.

In the AWS a lot of people inside any explanation, and a lot of auto-encoders have a fixttple window in machine learning or something like that to
2015-07-17 21:10:21,064 - INFO - Iteration 9600: 1.08303212064
. The Data Science are the same thing.  I'm just making a comprehensive topic, I was wondering why the samples are trained network degrees of AI and I
2015-07-17 21:12:16,700 - INFO - Iteration 9700: 1.07953141292
...  I don't have a more starting tool, what would that work teaching to achieve a lot of parameters?   If you correct any last week. All of the fun t
2015-07-17 21:14:10,635 - INFO - Iteration 9800: 1.07772277664
. Any probability of research community's progress in a statistical method produce a probabilities. It's also providing basic one-hot to put at any ot
2015-07-17 21:16:05,185 - INFO - Iteration 9900: 1.07404065218
. I was just thinking the first package for you to solve some pieces after a many competition.

Aal the predictions in the first set is that we're loo
'''
