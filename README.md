# U-DIT
## **Introduction**
This repository contains a new deep learning method based on vision transformer (ViT) to automatically localize and segment left ventricle (LV), right ventricle (RV) and myocardium (MYO) in cardiac MR images. The method is introduced in the following paper:
"[U-Dit: Unet-Like Dilated Transformer for Cardiac MRI Segmentation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4866882)"
## **Train the model:**
1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.To preprocess data and train the model run the script main.py.
```
python main.py --main-path your DATA_DIR
```
