import torch
import logging
import sys
import os
import random
import numpy as np
import shutil
import glob
import nibabel as nib
from skimage import transform
import argparse

from preprocess import image_utils

from networks.vision_transformer import LDTUnet as ViT_seg


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


def change_size(patient, imgs, imgs_big, pat_list, spec, change):
    ind = pat_list.index(patient)
    str = spec[ind]
    str2 = str.split('[')
    str3 = str2[-1].split(']')
    cu_spec = str3[0].split(',')
    x_new_var, y_new_var, w_new_var, h_new_var = int(cu_spec[0]), int(cu_spec[1]), int(cu_spec[2]), int(cu_spec[3])
    if change == 'small':
        imgs_crop = imgs[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var, :]
        return imgs_crop
    if change == 'big':
        imgs_big[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var, :] = imgs
        return imgs_big


def predict(main_path, test_name, save_path, model, target_resolution=(1.25, 1.25)):

    img_rows, img_cols = 224, 224

    for patient in test_name:
        folder_path = os.path.join(main_path, patient)

        infos = {}
        for line in open(os.path.join(folder_path, 'Info.cfg')):
            label, value = line.split(':')
            infos[label] = value.rstrip('\n').lstrip(' ')

        systole_frame = int(infos['ES'])
        diastole_frame = int(infos['ED'])

        if os.path.exists(os.path.join(save_path + '/' + patient)):
            shutil.rmtree(os.path.join(save_path + '/' + patient))
        os.makedirs(os.path.join(save_path + '/' + patient))

        for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
            file_base = file.split('.nii')[0]
            frame = int(file_base.split('frame')[-1])

            img_nii = nib.load(file)
            img_dat = img_nii.get_fdata()
            img = img_dat.copy()

            head = img_nii.header
            pixel_size = (head.structarr['pixdim'][1],
                          head.structarr['pixdim'][2])

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1], 1]
            slice_rescaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               mode='constant')

            img_cropped = image_utils.crop_or_pad_to_size(slice_rescaled, img_rows, img_cols)

            img_cropped2 = (img_cropped - np.min(img_cropped)) * 255 / (np.max(img_cropped) - np.min(img_cropped))

            for z in range(img_cropped.shape[-1]):
                if img_cropped[:, :, z].min() > 0:
                    img_cropped[:, :, z] -= img_cropped[:, :, z].min()

                img_tmp = img_cropped[:, :, z]

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma + 1e-10)
                img_cropped2[:, :, z] = img_tmp

            imgs_test = np.array(img_cropped2, np.float32)
            imgs_test_tot = np.zeros(imgs_test.shape, np.float32)

            for j in range(imgs_test.shape[-1]):
                img_exp1 = np.expand_dims(imgs_test[:, :, j], -1)
                img_exp = np.expand_dims(img_exp1, 0)
                img_test_fin = torch.from_numpy(img_exp).float()
                img_test_fin2 = img_test_fin.permute(3, 0, 1, 2).cuda()
                model.eval()
                with torch.no_grad():
                    pre = model(img_test_fin2)

                    _, pred = torch.max(pre, dim=1)

                    imgs_mask_test_final = pred.squeeze(0).cpu().numpy()

                    imgs_test_tot[:, :, j] = imgs_mask_test_final

            mask_real_size = image_utils.crop_or_pad_to_size(imgs_test_tot, slice_rescaled.shape[0],
                                                             slice_rescaled.shape[1])
            scale_vector = [target_resolution[0] / pixel_size[0], target_resolution[1] / pixel_size[1], 1]
            mask_real_scale = transform.rescale(mask_real_size,
                                                scale_vector,
                                                order=0,
                                                preserve_range=True,
                                                mode='constant',
                                                anti_aliasing=False)

            mask_real_scale = mask_real_scale.astype('uint8')
            if img.shape != mask_real_scale.shape:
                raise ValueError('the shape of image and mask is not equal')
            if frame == systole_frame:
                nimg = nib.Nifti1Image(mask_real_scale, affine=img_nii.affine, header=img_nii.header)
                nib.save(nimg,
                         os.path.join(save_path + '/' + patient + '/' + '%s_ES.nii.gz' % patient))

            elif frame == diastole_frame:
                nimg = nib.Nifti1Image(mask_real_scale, affine=img_nii.affine, header=img_nii.header)
                nib.save(nimg,
                         os.path.join(save_path + '/' + patient + '/' + '%s_ED.nii.gz' % patient))

        print(patient, 'prediction done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-path', type=str,
                        default='./dataset')
    parser.add_argument('--processed-root', type=str,
                        default='./processed_ACDC', help='root dir for test volume data')
    parser.add_argument('--dataset', type=str,
                        default='ACDC', help='experiment_name')
    parser.add_argument('--save-dir', type=str, help='saved model dir', default='./ckpt')
    parser.add_argument('--output-dir', type=str, help='output dir', default='./Prediction')
    parser.add_argument('--base-lr', type=float, default=0.05, help='segmentation network learning rate')
    parser.add_argument('--num-classes', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--img-size', type=int,
                        default=224, help='input patch size of network input')

    args = parser.parse_args()
    net = ViT_seg(img_size=args.img_size, num_classes=args.num_classes).cuda()
    snapshot = os.path.join(args.save_dir, 'best_model.pth')
    checkpoint = torch.load(snapshot)
    msg = net.load_state_dict(checkpoint['model_state_dict'])
    print("epoch number is:", checkpoint['epoch'])
    print("self trained swin unet", msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    _, _, test_data_name = real_arrange(args.main_path)

    predict(args.main_path, test_data_name, args.output_dir, net, target_resolution=(1.25, 1.25))
