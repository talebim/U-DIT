# U-DIT
## **Introduction**
This repository contains a new deep learning method based on vision transformer (ViT) to automatically localize and segment left ventricle (LV), right ventricle (RV) and myocardium (MYO) in cardiac MR images. The method is introduced in the following paper:
"[U-Dit: Unet-Like Dilated Transformer for Cardiac MRI Segmentation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4866882)"
## **Train the model:**
1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.To preprocess data and train the model run the script main.py.
```
python main.py --main-path your DATA_DIR --save-dir your save model DIR
```
The segmented image of test set will be saved in outputs.
## **Steps to test the pretrained model:**
1.To reproduce the results, download weights of our best model from **[here](https://drive.google.com/drive/u/0/folders/1nvVeGaBRPVT2r9oLPlpjhzYKAptHRE8D)**
 
2.Put the best_model.pth file in your save model DIR

2.Run the script predict.py.
```
python predict.py --processed-root your Processed DATA_DIR --save-dir your save model DIR --output-dir your OUT_DIR
```
## **Requirements**
The code is tested on Ubuntu 20.04 with the following components:

Software
Python 3.8
pytorch 1.13
CUDA 11.8 

## Logs
To launch the tensorboard instance run
```
tensorboard --logdir 'logs/U_DIT'
```
It will give a view on the evolution of the loss for both the training and validation data.


