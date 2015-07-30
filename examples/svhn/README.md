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


