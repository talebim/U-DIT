import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vision_transformer import LDTUnet as ViT_seg
from Dataload.trainer_ACDC import trainer_ACDC
from preprocess.preprocess_data import Preprocess
from predict import predict, real_arrange

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--main-path', type=str,
                    default='./dataset')
parser.add_argument('--processed-root', type=str,
                    default='./processed_ACDC')
parser.add_argument('--num-classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--save-dir', type=str, help='save model dir', default='./ckpt')
parser.add_argument('--output-dir', type=str, help='output dir', default='./Prediction')
parser.add_argument('--max-epochs', type=int,
                    default=5, help='maximum epoch number to train')
parser.add_argument('--batch-size', type=int,
                    default=5, help='batch_size per gpu')
parser.add_argument('--base-lr', type=float, default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img-size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--pretrain', type=str, default='./pretrained_ckpt/swin_tiny_patch4_window7_224.pth',
                    metavar="FILE", help='path to load pretrain file', )

args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    Preprocess(args)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    net = ViT_seg(img_size=args.img_size, num_classes=args.num_classes, pretrain_path=args.pretrain).cuda()

    cnt = count_parameters(net)
    print('model parametrs:', cnt)

    net.load_from(args.pretrain)

    trainer_ACDC(args, net, args.save_dir)

    _, _, test_data_name = real_arrange(args.main_path)

    predict(args.main_path, test_data_name, args.output_dir, net, target_resolution=(1.25, 1.25))
