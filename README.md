# U-DIT
## **Introduction**
This repository contains a new deep learning method based on vision transformer (ViT) to automatically localize and segment left ventricle (LV), right ventricle (RV) and myocardium (MYO) in cardiac MR images. The method is introduced in the following paper:
"[U-Dit: Unet-Like Dilated Transformer for Cardiac MRI Segmentation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4866882)"
## **Train the model:**
1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.To preprocess data and train the model run the script main.py.
```
python main.py --main-path your DATA_DIR --save-dir your OUT_DIR
```
The segmented image of test set will be saved in outputs.
## **Steps to test the pretrained model:**
1.To reproduce the results, download weights of our best model from **[here](https://drive.google.com/file/d/1iMSjN4b1y_uBoCqYYazqd33tP7uWjvCq/view?usp=drive_link)**
 
2.Put the last.ckpt file in ckpt folder

2.Run the script predict.py.
```
python predict.py --data-root your DATA_DIR --save-path your OUT_DIR
```



https://drive.google.com/drive/u/0/folders/1nvVeGaBRPVT2r9oLPlpjhzYKAptHRE8D
