import numpy as np

import apollo
from apollo import layers
from apollo.models import alexnet

from hyper import hyper
from util import Rect, show_rect_list, reduced_rect_list

class OverfeatNet:
    def __init__(self, net, batch):
        """
        Wrapper around Apollo network providing 
        convenience functions for passing / returning
        data.
        Must provide batch to test the network's 
        pass and establish layers.
        """
        self.net = net
        self.train_batch_is(batch)
        self.image_height = len(batch.image_array[0][0])
        self.image_width = len(batch.image_array[0][0][0])
        self.net.reset_forward()

    def batch_size(self):
        return len(self.batch.image_array)

    def param_update(self, lr):
        self.net.backward()
        self.net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])

    def test_batch_is(self, batch):
        val = self.batch_is(batch, "TEST")
        self.net.reset_forward()
        return val
        
    def train_batch_is(self, batch):
        val = self.batch_is(batch, "TRAIN")
        return val

    def batch_is(self, batch, phase):
        self.batch = batch
        image_array = batch.image_array
        bbox_label = batch.bbox_label_array
        conf_label = batch.conf_label_array
        binary_label_array = batch.binary_label_array
        net = self.net
        net.forward_layer(layers.NumpyData(name="data", data=image_array))
        net.forward_layer(layers.NumpyData(name="bbox_label", data=bbox_label))
        net.forward_layer(layers.NumpyData(name="conf_label", data=conf_label))
        net.forward_layer(layers.NumpyData(name="binary_label", data=binary_label_array))

        # parameters
        weight_filler = layers.Filler(type="xavier")
        bias_filler = layers.Filler(type="constant", value=0.2)
        conv_lr_mults = [1.0, 2.0]
        conv_decay_mults = [1.0, 0.0]

        # bunch of conv / relu layers
        net.forward_layer(layers.Convolution(name="conv1", bottoms=["data"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150))
        net.forward_layer(layers.ReLU(name="relu1", bottoms=["conv1"], tops=["conv1"]))

        net.forward_layer(layers.Convolution(name="conv2", bottoms=["conv1"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150))
        net.forward_layer(layers.ReLU(name="relu2", bottoms=["conv2"], tops=["conv2"]))

        # only pooling layer
        net.forward_layer(layers.Pooling(name="pool4", bottoms=["conv2"], kernel_size=2, stride=2)) # finished 2nd pool

        net.forward_layer(layers.Convolution(name="conv3", bottoms=["pool4"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150))
        net.forward_layer(layers.ReLU(name="relu3", bottoms=["conv3"], tops=["conv3"]))

        net.forward_layer(layers.Convolution(name="conv4", bottoms=["conv3"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150))
        net.forward_layer(layers.ReLU(name="relu4", bottoms=["conv4"], tops=["conv4"]))

        net.forward_layer(layers.Convolution(name="conv5", bottoms=["conv4"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150)) 
        net.forward_layer(layers.ReLU(name="relu5", bottoms=["conv5"], tops=["conv5"]))

        net.forward_layer(layers.Convolution(name="conv6", bottoms=["conv5"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150)) 
        net.forward_layer(layers.ReLU(name="relu6", bottoms=["conv6"], tops=["conv6"]))

        net.forward_layer(layers.Convolution(name="conv_0", bottoms=["conv6"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150)) 
        net.forward_layer(layers.ReLU(name="relu_0", bottoms=["conv_0"], tops=["conv_0"]))

        net.forward_layer(layers.Convolution(name="conv_1", bottoms=["conv_0"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150)) 
        net.forward_layer(layers.ReLU(name="relu_1", bottoms=["conv_1"], tops=["conv_1"]))

        net.forward_layer(layers.Convolution(name="conv_2", bottoms=["conv_1"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, stride=1, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=150)) 
        net.forward_layer(layers.ReLU(name="relu_2", bottoms=["conv_2"], tops=["conv_2"]))

        # inner product layers
        net.forward_layer(layers.Convolution(name="conv8", bottoms=["conv_2"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=1,
            weight_filler=weight_filler, bias_filler=bias_filler, num_output=4000))
        net.forward_layer(layers.ReLU(name="relu8", bottoms=["conv8"], tops=["conv8"]))
        layers.Dropout(name="drop0", bottoms=["conv8"], tops=["conv8"], dropout_ratio=0.5, phase=phase),
        net.forward_layer(layers.Convolution(name="L7", bottoms=["conv8"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=1,
            weight_filler=weight_filler, bias_filler=bias_filler, num_output=4000))
        net.forward_layer(layers.ReLU(name="relu9", bottoms=["L7"], tops=["L7"]))
        layers.Dropout(name="drop1", bottoms=["L7"], tops=["L7"], dropout_ratio=0.5, phase=phase),

        # binary prediction layers: is a character here ? 
        net.forward_layer(layers.Convolution(name="binary_conf_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=2))
        binary_softmax_loss = net.forward_layer(layers.SoftmaxWithLoss(name='binary_softmax_loss', bottoms=['binary_conf_pred', 'binary_label']))
        net.forward_layer(layers.Softmax(name='binary_softmax', bottoms=['binary_conf_pred']))

        # character predictions
        label_softmax_loss = 0
        net.forward_layer(layers.Convolution(name="label_conf_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=11))
        label_softmax_loss = net.forward_layer(layers.SoftmaxWithLoss(name='label_softmax_loss', bottoms=['label_conf_pred', 'conf_label'], loss_weight=1., ignore_label = 0))
        net.forward_layer(layers.Softmax(name='label_softmax', bottoms=['label_conf_pred']))

        # bounding box prediction
        net.forward_layer(layers.Convolution(name="bbox_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults,
            param_decay_mults=[0., 0.], kernel_size=1,
            weight_filler=weight_filler, bias_filler=bias_filler, num_output=4))
        net.forward_layer(layers.Concat(name='bbox_mask', bottoms =  4 * ['binary_label']))
        net.forward_layer(layers.Eltwise(name='bbox_pred_masked', bottoms=['bbox_pred', 'bbox_mask'], operation='PROD'))
        net.forward_layer(layers.Eltwise(name='bbox_label_masked', bottoms=['bbox_label', 'bbox_mask'], operation='PROD'))
        bbox_loss = net.forward_layer(layers.L1Loss(name='l1_loss', bottoms=['bbox_pred_masked', 'bbox_label_masked'], loss_weight=0.001))
        
        return (binary_softmax_loss, label_softmax_loss, bbox_loss)

    # returns list of bounding boxes using non max suppresssion
    def __init_rect_list(self, ind, min_prob = 0.5):   
        """
        Returns list of Rect instances from output of neural network for a single image.

        Args
            bottom_height (int): height of image (bottom layer)
            bottom_width (int): width of image
            bbox_label_pred (ndarray): spatial map of predicted bounding 
                box coordinates at the top of neural network
            label_pred (ndaray): spatial map of probabilities of the 
                different labels
        
        """
        #bbox_label_pred = self.net.tops['bbox_label'].data[ind]
        #binary_pred = self.net.tops['binary_label'].data[ind]
        bottom_height = self.image_height
        bottom_width = self.image_width
        bbox_label_pred = self.net.tops['bbox_pred'].data[ind]
        binary_pred = self.net.tops['binary_softmax'].data[ind]
        label_pred = self.net.tops['label_softmax'].data[ind]
        
        (_, top_height, top_width) = bbox_label_pred.shape
        y_mul = bottom_height * 1. / top_height
        x_mul = bottom_width * 1. / top_width
        rect_list = []
        for y in xrange(top_height):
            for x in xrange(top_width):
                # corresponds to indices in original image
                cx_orig = x_mul * (x + 0.5)
                cy_orig = y_mul * (y + 0.5)

                # we predict a symbol here
                if binary_pred[0, y, x] < 0.5:
                    k = np.argmax(label_pred[:, y, x]) 
                    #if label_pred[k, y, x] < 0.2: continue

                    # apply offsets to get positions in original image
                    cx = cx_orig + bbox_label_pred[0, y, x]
                    cy = cy_orig + bbox_label_pred[1, y, x]
                    w = bbox_label_pred[2, y, x]
                    h = bbox_label_pred[3, y, x]
                    xmin = cx - w / 2.0
                    ymin = cy - h / 2.0
                    rect = Rect(xmin, ymin, xmin + w, ymin + h, label=k, prob=label_pred[k, y, x])
                    rect_list.append(rect)

        return rect_list

    def rect_list(self, ind):
        """
        Args:
            ind (int): sample index corresponding to 
            index in batch.
        Returns:
            List(Rect): list of Rect instances 
            extracted from the top of the network.
        """
        rect_list = self.__init_rect_list(ind)
        rect_list = reduced_rect_list(rect_list)
        return rect_list


