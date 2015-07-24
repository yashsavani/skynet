import os
import random
import glob
import json

import numpy as np
import h5py
import cv2

random.seed(0)
np.random.seed(0)

apollo_root = os.environ['APOLLO_ROOT']

IMG_HEIGHT = 120.
IMG_WIDTH = 192.
TOP_HEIGHT = 4
TOP_WIDTH = 6
#IMG_HEIGHT = 480.
#IMG_WIDTH = 640.
#TOP_HEIGHT = 15
#TOP_WIDTH = 20

# one for each digit, plus one label for no character here
LABEL_CT = 11

class Rect:
    def __init__(self, cy, cx, h, w, label = None):
        self.cx = cx
        self.cy = cy
        self.w = w 
        self.h = h
        self.label = label

    def intersects(self, other):
        w_intersects = (self.w + other.w) / 2 > abs(self.cx - other.cx) 
        h_intersects = (self.h + other.h) / 2 > abs(self.cy - other.cy) 
        return w_intersects and h_intersects

class Datum:
    def __init__(self, image, box_list, normalized=False):
        self.image = image
        # one hot array of labels
        label_array = np.zeros(len(box_list))
        # array corresponds to cy, cx, h, w
        bbox_array = np.zeros((len(box_list), 4))
        for ind, box in enumerate(box_list):
            # zero label means nothing is there
            label_array[ind] = int(box["label"])
            bbox_array[ind][0] = box["top"] + box["height"] // 2
            bbox_array[ind][1] = box["left"] + box["width"] // 2
            bbox_array[ind][2] = box["height"]
            bbox_array[ind][3] = box["width"]

        self.bbox_array = bbox_array
        self.label_array = label_array
        self.normalized = normalized

    def rect_iter(self):
        #for elem in self.bbox_array:
        for ind in xrange(len(self.bbox_array)):
            
            yield Rect(self.bbox_array[ind][0],  
                      self.bbox_array[ind][1], 
                      self.bbox_array[ind][2], 
                      self.bbox_array[ind][3], 
                      self.label_array[ind])

    def prepare_data(self):
        conf_label = np.zeros((1, TOP_HEIGHT, TOP_WIDTH), dtype = np.float)
        bbox_label = np.zeros((4, TOP_HEIGHT, TOP_WIDTH), dtype = np.float)

        for i in range(TOP_HEIGHT):
            for j in range(TOP_WIDTH):
                for rect in self.rect_iter():
                    obs_rect = Rect(16 + 32 * i, 16 + 32 * j, 64, 64)
                    if rect.intersects(obs_rect):
                        #conf_label[:, i, j] = rect.label
                        conf_label[0, i, j] = 1
                        bbox_label[0, i, j] = rect.cy - obs_rect.cy
                        bbox_label[1, i, j] = rect.cx - obs_rect.cx
                        bbox_label[2, i, j] = rect.h 
                        bbox_label[3, i, j] = rect.w 
                        break

        trans_img = self.image.reshape(1, IMG_HEIGHT, IMG_WIDTH)
        return (trans_img, bbox_label, conf_label)

    def display_image(self, rgb_color=(0,255,0)):
        # returns image where rectangle is drawn on copy of image (greyscale or not).  
        image = np.copy(self.image)
        
        # if we have greyscale image, convert it to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            
        for bbox in self.bbox_array:
            ymin = int(bbox[0] - bbox[2] / 2)
            xmin = int(bbox[1] - bbox[3] / 2)
            ymax = int(ymin + bbox[2])
            xmax = int(xmin + bbox[3])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rgb_color, 2)
	
        return image

    def is_normalized(self):
        if self.normalized:  
            return 

        # put image on white canvas with proper aspect ratio
        (h, w) = self.image.shape
        h_mul = IMG_HEIGHT / h
        w_mul = IMG_WIDTH / w
        mul = min(h_mul, w_mul)

        # resize image and bounding boxes
        (h_new, w_new) = (int(mul * h), int(mul * w))
        self.bbox_array *= mul
        image_resized = cv2.resize(self.image, (w_new, h_new))

        # apply offsets so that image lies in the center
        y_offset = (IMG_HEIGHT - h_new) // 2
        x_offset = (IMG_WIDTH - w_new) // 2
        image_new = 255 * np.ones((IMG_HEIGHT, IMG_WIDTH)).astype(np.uint8)
        image_new[y_offset:y_offset + h_new, x_offset:x_offset + w_new] = image_resized

        # set attributes
        self.image = image_new
        self.bbox_array[:,0] += y_offset
        self.bbox_array[:,1] += x_offset
        self.normalized = True

# data_type is either "train" or "test"
def get_datum_iterator(data_type="train"):
    data_source = '%s/data/svnh/' % apollo_root
    images_dir = os.path.join(data_source, data_type)
    digit_struct_path = os.path.join(data_source, '%s.json' % data_type)
    with open(digit_struct_path, 'r') as f:
        digit_struct = json.load(f)

    while True:
        elem = random.choice(digit_struct)
        box_list = elem['boxes']
        file_path= os.path.join(images_dir, elem['filename'])
        image = cv2.imread(file_path, 0)
        datum = Datum(image, box_list)
        datum.is_normalized()
        yield datum

def get_batch_iterator(batch_size):
    datum_iterator = get_datum_iterator()
    while True:
        trans_img_list = []
        bbox_label_list = []
        conf_label_list = []
        for i in xrange(batch_size):
            datum = next(datum_iterator)
            trans_img, bbox_label, conf_label = datum.prepare_data()
            trans_img_list.append(trans_img)
            bbox_label_list.append(bbox_label)
            conf_label_list.append(conf_label)

        batch = (np.array(trans_img_list),
                 np.array(bbox_label_list),
                 np.array(conf_label_list),
                 )
        yield batch

for datum in get_datum_iterator():
    image = datum.display_image()
    cv2.imwrite("test.png", image)
    break

