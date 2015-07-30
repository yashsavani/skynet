import numpy as np 
import cv2

from nms import nms_detections

class Rect:
    def __init__(self, xmin, ymin, xmax, ymax, label = None, prob = None):
        """
        Container class for a bounding box in an image.  
        At the minimum, contains box coordinates, optionally holds
        label and probability information as well (useful for non-max
        suppression).
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # center coordinates
        self.cx = (self.xmin + self.xmax) / 2
        self.cy = (self.ymin + self.ymax) / 2
        # width and height
        self.w = self.xmax - self.xmin
        self.h = self.ymax - self.ymin
        # optional params
        self.label = label
        self.prob = prob

    def scale_wh(self, w_scale, h_scale):
        """
        Keeps center coordinates of bounding box 
        but scales box dimensions by a factor by alpha.
        """
        self.w *= w_scale
        self.h *= h_scale
        self.xmin = self.cx - self.w / 2
        self.ymin = self.cy - self.h / 2
        self.xmax = self.xmin + self.w
        self.ymax = self.ymin + self.h

    def intersects(self, other):
        w_intersects = (self.w + other.w) / 2 > abs(self.cx - other.cx) 
        h_intersects = (self.h + other.h) / 2 > abs(self.cy - other.cy) 
        return w_intersects and h_intersects

def show_rect_list(image, rect_list, rgb_color=(0,255,0)):
    """
    Returns image where rectangle list is drawn on copy of image (greyscale or not).  

    Args:
        image (ndarray): image data
        rect_list (list): list of Rect instances we want to display
        rgb_color (tuple): color we want to use to draw rectangles
    """
    image = np.copy(image)
    
    # if we have greyscale image, convert it to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        
    for rect in rect_list:
        # make sure box fits in image
        ymin = max(int(rect.ymin), 0)
        xmin = max(int(rect.xmin), 0)
        ymax = min(int(rect.ymax), image.shape[0] - 1)
        xmax = min(int(rect.xmax), image.shape[1] - 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rgb_color, 2)

    return image

def reduced_rect_list(rect_list, max_len = 5):
    # performs non max suppresion on bounding box list
    detection_array = np.zeros((len(rect_list), 6))
    for i, rect in enumerate(rect_list):
        detection_array[i][0] = rect.xmin
        detection_array[i][1] = rect.ymin
        detection_array[i][2] = rect.xmax
        detection_array[i][3] = rect.ymax
        detection_array[i][4] = rect.prob
        detection_array[i][5] = rect.label

    detection_array_new = nms_detections(detection_array, 0.3)
    rect_list_new = []
    for row in detection_array_new:
        rect_new = Rect(row[0], row[1], row[2], row[3], prob = row[4], label = row[5])
        rect_new.scale_wh(2., 2.)
        rect_list_new.append(rect_new)
        if len(rect_list_new) == max_len: 
            break

    return rect_list_new

