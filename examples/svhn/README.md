# OverFeat with SVHN

This example trains an OverFeat neural network to detect and localize house numbers.  

The dataset consists of over 200,000 images of house numbers gathered by Google street view, as well as json files containing bounding box and label information.  See [http://ufldl.stanford.edu/housenumbers/](here) for more information about SVHN.  

To run the example, you must first download the dataset: 
```
cd $APOLLO_ROOT
./data/svhn/get_svhn.sh
```
The dataset is large so this will take a few minutes.  Once the download is done, simply:
```
cd $APOLLO_ROOT/examples/svhn
python train.py
```
The loss is logged to the console.  Predictions are saved in the `test` subdirectory. 

After a few thousand iterations the network starts making reasonable predictions:

![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_0.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_1.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_2.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_4.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_5.png)

The predicted symbols are written in red at the top and correspond to the bounding boxes as ordered from left to right.

### overfeat_net.py

Contains the network structure for the neural network that is used.  The network consists of repeated convolutional / relu layers.  At the top of the network is where things are a bit different:

```python
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
```

Since we want a spatial map at a downsampled resolution on top, we do a convolution with kernel of 1 instead of a fully connected layer as is typically the case for image classification.  The result of these layers is a feature vector for each region in the image.  We then put these features to work:

```python
# binary prediction layers: is a character here ? 
net.forward_layer(layers.Convolution(name="binary_conf_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=2))
binary_softmax_loss = net.forward_layer(layers.SoftmaxWithLoss(name='binary_softmax_loss', bottoms=['binary_conf_pred', 'binary_label']))
net.forward_layer(layers.Softmax(name='binary_softmax', bottoms=['binary_conf_pred']))
```

This layer creates a softmax binary classifier at each point in the spatial map that will predict whether a character exists at this point.  We also want to know the identity of the characters, which we do separately as follows:

```python
# character predictions
net.forward_layer(layers.Convolution(name="label_conf_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=11))
label_softmax_loss = net.forward_layer(layers.SoftmaxWithLoss(name='label_softmax_loss', bottoms=['label_conf_pred', 'conf_label'], loss_weight=1., ignore_label = 0))
net.forward_layer(layers.Softmax(name='label_softmax', bottoms=['label_conf_pred']))
```

The label 0 corresponds to the background class (SVHN uses label 10 for the number 0).  By ignoring this label, this softmax layer can focus on predicting labels at different spatial locations without having to worry about predicting the background class.  Finally, we add a layer to do bounding box regression: 

```python
# bounding box prediction
net.forward_layer(layers.Convolution(name="bbox_pred", bottoms=["L7"], param_lr_mults=conv_lr_mults,
    param_decay_mults=[0., 0.], kernel_size=1,
    weight_filler=weight_filler, bias_filler=bias_filler, num_output=4))
net.forward_layer(layers.Concat(name='bbox_mask', bottoms =  4 * ['binary_label']))
net.forward_layer(layers.Eltwise(name='bbox_pred_masked', bottoms=['bbox_pred', 'bbox_mask'], operation='PROD'))
net.forward_layer(layers.Eltwise(name='bbox_label_masked', bottoms=['bbox_label', 'bbox_mask'], operation='PROD'))
bbox_loss = net.forward_layer(layers.L1Loss(name='l1_loss', bottoms=['bbox_pred_masked', 'bbox_label_masked'], loss_weight=0.001))
```

Note that each point in the spatial map corresponds to a rectangular region in the original image.  To do bounding box regression, we estimate the cx and cy offsets (center x, y coordinates) of the bounding box around the object relative to the center pixel of the corresponding rectangular region.  We also estimate the width and the height of the bounding box.  The regression layer creates a spatial map `map` of depth 4 where:  
    * map[0, y, x] - cx offset at (x,y) position
    * map[1, y, x] - cy offset at (x,y) position
    * map[2, y, x] - width of bounding box
    * map[3, y, x] - height of bounding box

We use a mask so that the we don't penalize the network for making bad predictions at locations in the image where no object is present.  

The network will predict multiple bounding boxes for each object.  Use use non-max suppression to filter these boxes.  We note that this is an imperfect process.  A common issue is predicting multiple boxes for a single object instance. 

