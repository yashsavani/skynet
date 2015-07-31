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
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_3.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_4.png)
![](https://raw.githubusercontent.com/Russell91/apollo/master/examples/svhn/images/pred_5.png)

The predicted symbols are written in red at the top and correspond to the bounding boxes as ordered from left to right.


