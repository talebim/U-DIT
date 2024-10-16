# U-DIT
## **Introduction**
This repository contains a new deep learning method based on vision transformer (ViT) to automatically localize and segment the left ventricle (LV), right ventricle (RV), and myocardium (MYO) in cardiac MR images. The method is introduced in the following paper:
"[U-DIT: Unet-like Dilated Transformer for Cardiac MRI Segmentation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4866882)"
## **Train the model:**
1. Register and download the ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html.

2. Download pretrained weights from [Swin-Unet github](https://github.com/HuCaoFighting/Swin-Unet) and put it in a folder named pretrained_ckpt.

3. To preprocess data and train the model run the script main.py.
```
python main.py --main-path your DATA_DIR --save-dir your save model DIR
```
The segmented image of the test set will be saved in outputs.
## **Steps to test the pre-trained model:**
1. To reproduce the results, download the weights of our best model from **[here](https://drive.google.com/drive/u/0/folders/1nvVeGaBRPVT2r9oLPlpjhzYKAptHRE8D)**
 
2. Put the best_model.pth file in your save model DIR

2. Run the script predict.py.
```
python predict.py --main-path your DATA_DIR --save-dir your save model DIR --output-dir your OUT_DIR
```
## **Requirements**
The code is tested on Ubuntu 20.04 with the following components:

Software
Python 3.8
Pytorch 2.3.1
CUDA 12.4

## Logs
To launch the tensorboard instance run
```
tensorboard --logdir 'logs/U_DIT'
```
It will give a view of the evolution of the loss for both the training and validation data.


