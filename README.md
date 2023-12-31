# U-DIT: Unet-like Dilated Transformer for Cardiac MRI Segmentation
## **Introduction**
This repository proposes a new deep learning method based on vision transformer (ViT) to automatically localize and segment left ventricle (LV), right ventricle (RV) and myocardium (MYO) in cardiac MR images. The proposed structure is a U-shape transformer model inspired from Unet in CNN models. The model utilized global attention blocks and local-dilated (LD) blocks in its structure. The global attention blocks extract long-range dependencies in shallow layers, while the LD blocks extract local features that cannot be extracted with classical blocks of ViTs. 

