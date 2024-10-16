import os
import shutil
import glob
import numpy as np
import random
import csv
import pydicom
import cv2
import nibabel as nib
import logging
from skimage import transform

from .image_utils import crop_or_pad_to_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

img_rows, img_cols = 224, 224
# target_resolution = (1.36719, 1.36719)
target_resolution = (1.25, 1.25)


def creat_folds(dirpath):
    if os.path.exists(os.path.join(dirpath)):
        shutil.rmtree(os.path.join(dirpath))
    os.makedirs(os.path.join(dirpath))

    os.makedirs(dirpath + '/' + 'val')
    os.makedirs(dirpath + '/' + 'mask_val')

    os.makedirs(dirpath + '/' + 'train')
    os.makedirs(dirpath + '/' + 'mask_train')

    os.makedirs(dirpath + '/' + 'test')
    # os.makedirs(dirpath + '/' + 'mask_test')

    return dirpath


def real_arrange(path):
    file_tot = sorted(os.listdir(path))[:100]
    random.seed(4)
    file_tot_rnd = random.sample(file_tot, len(file_tot))
    nb_data = len(file_tot_rnd)
    nb_val = round(0.15 * nb_data)
    val_data_name = file_tot_rnd[:nb_val]
    train_data_name = file_tot_rnd[nb_val:]

    test_data_name = sorted(os.listdir(path))[100:150]

    return train_data_name, val_data_name, test_data_name


def process_ACDC(img_path, dirpath, case, data_type):
    folder_path = os.path.join(img_path, case)

    infos = {}
    for line in open(os.path.join(folder_path, 'Info.cfg')):
        label, value = line.split(':')
        infos[label] = value.rstrip('\n').lstrip(' ')

    for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
        file_base = file.split('.nii')[0]
        file_mask = file_base + '_gt.nii.gz'
        img_name = file_base.split('/')[-1]

        print('image is for the case:', img_name)
        img_nii = nib.load(file)
        mask_nii = nib.load(file_mask)

        img_dat = img_nii.get_fdata()
        mask_dat = mask_nii.get_fdata()

        img = img_dat.copy()
        mask = mask_dat.copy()

        pixel_size = (img_nii.header.structarr['pixdim'][1],
                      img_nii.header.structarr['pixdim'][2])

        scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1], 1]

        slice_rescaled = transform.rescale(img,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           mode='constant')

        mask_rescaled = transform.rescale(mask,
                                          scale_vector,
                                          order=0,
                                          preserve_range=True,
                                          anti_aliasing=False,
                                          mode='constant')

        slice_cropped = crop_or_pad_to_size(slice_rescaled, img_rows, img_cols)
        mask_cropped = crop_or_pad_to_size(mask_rescaled, img_rows, img_cols)

        slice_cropped2 = np.zeros(slice_cropped.shape)

        for z in range(slice_cropped.shape[-1]):
            if slice_cropped[:, :, z].min() > 0:
                slice_cropped[:, :, z] -= slice_cropped[:, :, z].min()

            img_tmp = slice_cropped[:, :, z]

            mu = img_tmp.mean()
            sigma = img_tmp.std()
            img_tmp = (img_tmp - mu) / (sigma + 1e-10)
            slice_cropped2[:, :, z] = img_tmp

        mask_cropped2 = np.round(mask_cropped).astype('uint8')
        if data_type != 'test':
            nimg = nib.Nifti1Image(mask_cropped2, affine=mask_nii.affine, header=mask_nii.header)
            nib.save(nimg,
                     os.path.join(dirpath, 'mask_' + '%s', '%s_mask.nii.gz') % (data_type, img_name))
        nimg = nib.Nifti1Image(slice_cropped2, affine=img_nii.affine, header=img_nii.header)
        nib.save(nimg, os.path.join(dirpath, '%s', '%s.nii.gz') % (data_type, img_name))


def preprocess_ACDC(img_path, dir_path, file_tot, data_type):
    for case in file_tot:
        process_ACDC(img_path, dir_path, case, data_type)


def Preprocess(args):
    train_data_name, val_data_name, test_data_name = real_arrange(args.main_path)

    loc_path = creat_folds(args.processed_root)

    preprocess_ACDC(args.main_path, loc_path, train_data_name, data_type='train')
    preprocess_ACDC(args.main_path, loc_path, val_data_name, data_type='val')
    preprocess_ACDC(args.main_path, loc_path, test_data_name, data_type='test')
