import os
import random
import glob
import json
import logging

import numpy as np
import h5py
import cv2

from util import Rect

random.seed(0)
np.random.seed(0)

apollo_root = os.environ['APOLLO_ROOT']

#IMG_HEIGHT = 120.
#IMG_WIDTH = 192.
IMG_HEIGHT = 60.
IMG_WIDTH = 96.
SUB_IMG_HEIGHT = IMG_HEIGHT * 1.
SUB_IMG_WIDTH = IMG_WIDTH * 1.
#TOP_HEIGHT = 15
#TOP_WIDTH = 24
TOP_HEIGHT = 30
TOP_WIDTH = 48


class Datum:
    def __init__(self, image, box_list, name = "datum", normalized=False):
        """
        Loads data container for labelled data for a single image.
        As input, takes greyscale image, and list of boxes 
        loaded from train.json and test.json in the 
        /data/svhn/ directory. 
        """
        self.image = image
        self.name = name
        # contains numerical indices of labels for each box
        label_array = np.zeros(len(box_list))
        bbox_array = np.zeros((len(box_list), 4))
        for ind, box in enumerate(box_list):
            label_array[ind] = int(box["label"])
            #each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
            bbox_array[ind][0] = box["left"] 
            bbox_array[ind][1] = box["top"]
            bbox_array[ind][2] = box["left"] + box["width"]
            bbox_array[ind][3] = box["top"] + box["height"]

        self.bbox_array = bbox_array
        self.label_array = label_array
        self.normalized = normalized

    @property
    def rect_list(self):
        """
        List[Rect] - instances corresponding to labelled image.
        """
        rect_list_ = []
        for ind in xrange(len(self.bbox_array)):
            rect = Rect(self.bbox_array[ind][0],  
                      self.bbox_array[ind][1], 
                      self.bbox_array[ind][2], 
                      self.bbox_array[ind][3], 
                      label = self.label_array[ind])

            rect_list_.append(rect)
        return rect_list_

    def rect_iter(self):
        for ind in xrange(len(self.bbox_array)):
            rect = Rect(self.bbox_array[ind][0],  
                      self.bbox_array[ind][1], 
                      self.bbox_array[ind][2], 
                      self.bbox_array[ind][3], 
                      label = self.label_array[ind])
            yield rect
                    
    def batch_data(self, image_mean):
        """
        Prepares data for ingestion by Apollo.

        Returns:
            ndarray (trans_img): transposed image in N-C-H-W caffe blob format
            ndarray (bbox_label): target bounding boxes corresponding to image, 
                bbox_label[0]: cx (center x) offset at each position
                bbox_label[1]: cy (center y) offset at each position
                bbox_label[2]: w (width) of bounding box of object at each position
                bbox_label[3]: h (height) of bounding box of object at each position
            ndarray (conf_label): target indices corresponding to different points in image
        """
        # zero label means nothing is there
        conf_label = np.zeros((1, TOP_HEIGHT, TOP_WIDTH), dtype = np.float)
        bbox_label = np.zeros((4, TOP_HEIGHT, TOP_WIDTH), dtype = np.float)
        y_mul = IMG_HEIGHT * 1. / TOP_HEIGHT
        x_mul = IMG_WIDTH * 1. / TOP_WIDTH

        # iterate over all pixels
        for y in range(TOP_HEIGHT):
            for x in range(TOP_WIDTH):
                # current distance to closest object center at coordinate
                best_dist = np.Inf
                # find object closest to coordinate (if one exists)
                for rect in self.rect_iter():
                    # makes box smaller to prevent label ambiguity 
                    # at downsampled resolution (detections get centered)
                    rect.scale_wh(0.5, 0.5)
                    x_orig = x_mul * x
                    y_orig = y_mul * y
                    obs_rect = Rect(x_orig, y_orig, x_orig + x_mul, y_orig + y_mul)
                    if rect.intersects(obs_rect):
                        center_dist = (rect.cx - obs_rect.cx) ** 2 + (rect.cy - obs_rect.cy) ** 2
                        if center_dist < best_dist:
                            best_dist = center_dist
                            conf_label[0, y, x] = rect.label
                            bbox_label[0, y, x] = rect.cx - obs_rect.cx
                            bbox_label[1, y, x] = rect.cy - obs_rect.cy
                            bbox_label[2, y, x] = rect.w
                            bbox_label[3, y, x] = rect.h

        image = self.image.copy()
        image = image.reshape(1, IMG_HEIGHT, IMG_WIDTH)

        return (image, bbox_label, conf_label)

    def is_normalized(self):
        """
        Resizes and translates images so that they are all centered inside 
        IMG_HEIGHT, IMG_WIDTH pixels.  Modifies bounding boxes accordingly.  
        """
        if self.normalized:  
            return 

        # put image on white canvas with proper aspect ratio
        (h, w) = self.image.shape
        h_mul = SUB_IMG_HEIGHT / h
        w_mul = SUB_IMG_WIDTH / w
        mul = min(h_mul, w_mul)

        # resize image and bounding boxes
        (h_new, w_new) = (int(mul * h), int(mul * w))
        self.bbox_array *= mul
        image_resized = cv2.resize(self.image, (w_new, h_new))

        # apply offsets so that image lies in the center
        x_offset = (IMG_WIDTH - w_new) // 2
        y_offset = (IMG_HEIGHT - h_new) // 2
        image_new = np.zeros((IMG_HEIGHT, IMG_WIDTH)).astype(np.uint8)
        image_new[y_offset:y_offset + h_new, x_offset:x_offset + w_new] = image_resized

        # save attributes
        self.image = image_new
        self.bbox_array[:,0] += x_offset
        self.bbox_array[:,1] += y_offset
        self.bbox_array[:,2] += x_offset
        self.bbox_array[:,3] += y_offset
        self.normalized = True

class Batch:    
    def __init__(self, image_mean):
        """
        Training batch appropriate for 
        apollo consumption.
        """ 
        self.datum_list = []
        self.image_list = []
        self.bbox_label_list = []
        self.conf_label_list = []
        self.binary_label_list = []
        self.image_mean = image_mean

    def datum_add(self, datum):
        trans_img, bbox_label, conf_label = datum.batch_data(self.image_mean)
        self.datum_list.append(datum)
        self.image_list.append(trans_img)
        self.bbox_label_list.append(bbox_label)
        self.conf_label_list.append(conf_label)
        binary_label = np.array(conf_label != 0, dtype = "f")
        self.binary_label_list.append(binary_label)

    @property
    def binary_label_array(self):
        #print self.binary_label_list[0]
        return np.array(self.binary_label_list)

    @property
    def bbox_label_array(self):
        return np.array(self.bbox_label_list)
        
    @property
    def conf_label_array(self):
        return np.array(self.conf_label_list)

    @property
    def image_array(self):
        return np.array(self.image_list)

def get_datum_iterator(data_type = "train", is_random = True):
    """
    Args:
        data_type (str): specifies training or testing data ("train" or "test")
        is_random (bool): if True randomly select each datum 
    Returns:
        Iter(Datum): iterator over datum instances.
    """
    data_source = '%s/data/svhn/' % apollo_root
    data_type = "test"
    #is_random = False
    #is_random = True
    is_random = False

    if data_type == "train":
        # combine training and extra training data
        digit_struct_path = os.path.join(data_source, "train.json")
        with open(digit_struct_path, 'r') as f:
            digit_struct = json.load(f)

        digit_struct_path = os.path.join(data_source, "extra.json")
        with open(digit_struct_path, 'r') as f:
            digit_struct += json.load(f)

    elif data_type == "test":
        digit_struct_path = os.path.join(data_source, "extra.json")
        #digit_struct_path = os.path.join(data_source, "test.json")
        with open(digit_struct_path, 'r') as f:
            digit_struct = json.load(f)

    ind = 0 
    while True and ind < len(digit_struct):
        if is_random:
            elem = random.choice(digit_struct)
        else:
            elem = digit_struct[ind]
            ind = ind + 1 
            ind = ind % len(digit_struct)

        box_list = elem['boxes']
        file_path = os.path.join(data_source, elem['filename'])
        image = cv2.imread(file_path, 0)
        if image == None: 
            logging.info("Warning: %s not found", file_path)
            continue

        basename = os.path.basename(elem['filename'])
        name = os.path.splitext(basename)[0]
        datum = Datum(image, box_list, name = name)
        datum.is_normalized()
        yield datum

def get_batch_iterator(batch_size, image_mean = None, data_type = "train", is_random = True):
    """
    Args:
        batch_size (int) - number of datum 
        elements contained by each batch instance.

    Returns:
        Iter(Batch) - iterator of batch instances for 
        consumption by Apollo.
    """
    datum_iterator = get_datum_iterator(data_type = data_type, is_random = is_random)
    while True:
        batch = Batch(image_mean = image_mean)
        for i in xrange(batch_size):
            datum = next(datum_iterator)
            batch.datum_add(datum)
        yield batch
