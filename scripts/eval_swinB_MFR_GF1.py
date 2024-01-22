import os
import random
import argparse
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm
from torch.utils import data
import torch.nn.functional as F

from datasets.gf1_dataset import GF1_FULL_SUP_condFEATS, GF1_cls_WEAKMFC
from metrics import StreamSegMetrics
from datasets import GF1_cls_WEAK, GF1_FULL_SUP, GF1_cls_WEAK_condFEATS
from guided_diffusion.swinB_MFR_parall import swin_MFR_Parall
import math

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='..dataset/GF1_datasets/datasets_321/data/',
                        help="path to Dataset")

    # Model Options
    parser.add_argument("--model", type=str, default='swinB_MFR_GF1',
                        choices=['HCDNet_cls_GF1', 'ResNet34_cls_GF1', 'mResNet50_cls_GF1'],
                        help='model name')
    parser.add_argument("--num_classes", type=int, default=2, help="num classes (default: None)")
    parser.add_argument("--gpu_id", type=str, default='5', help="GPU ID")

    # Train Options
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 4)')

    parser.add_argument("--ckpt", default='../checkpoints/swinB_MFR_GF1/best_swinB_MFR_GF1.pth',
                        help="restore from checkpoint")
    parser.add_argument("--save_path", default='..results/MFR_feats/',
                        help="restore from checkpoint")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")

    return parser

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'gf1':
        val_dst = GF1_FULL_SUP(root=opts.data_root, image_set='trainval')
    return val_dst


def validate(opts, model, loader, device, metrics_seg, metrics_gaus):
    """Do validation and return specified samples"""
    metrics_seg.reset()
    metrics_gaus.reset()

    right, forgot, alarm, all = 0, 0, 0, 0
    index=0

    with torch.no_grad():
        for i, (images, block_label, target, name) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)

            img_cls, seg_pre, cond_feat = model(images)

            # Evaluation
            seg_pre = seg_pre.squeeze()
            mean = torch.mean(seg_pre)
            std = torch.std(seg_pre)
            gaus_pre = GaussProjection(seg_pre, mean, std)

            pure_map = torch.ones_like(seg_pre)
            pure_map[seg_pre >= 0.8] = 0
            pure_map[seg_pre <= 0.2] = 0

            gaus_pre_mod = pure_map * gaus_pre
            sum_pre = (1-pure_map)*seg_pre + gaus_pre_mod

            seg_pre = seg_pre.cpu().numpy()
            sum_pre = sum_pre.cpu().numpy()
            masks = target.cpu().numpy()

            output = np.zeros((opts.batch_size, 320, 320), dtype=int)
            gaus_output = np.zeros((opts.batch_size, 320, 320), dtype=int)

            for b in range(seg_pre.shape[0]):
                norm_cam= seg_pre[b,] / np.max(seg_pre[b,])
                output[b,][norm_cam >= 0.4] = 1

                norm_cam_gaus = sum_pre[b,] / np.max(sum_pre[b,])
                gaus_output[b,][norm_cam_gaus >= 0.4] = 1

            metrics_seg.update(masks, output)
            metrics_gaus.update(masks, gaus_output)


        seg_score = metrics_seg.get_results()
        gaus_score = metrics_gaus.get_results()


        accuracy = right / all
        error = forgot / all
        false = alarm / all

    return accuracy, error, false, seg_score, gaus_score


def main():
    opts = get_argparser().parse_args()

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.batch_size,
                                 drop_last=True, pin_memory=False)
    print("Dataset: %s, Val set: %d" % (opts.dataset, len(val_dst)))

    # Set up metrics
    metrics_seg = StreamSegMetrics(opts.num_classes)
    metrics_gaus = StreamSegMetrics(opts.num_classes)

    model = swin_MFR_Parall(in_chans=4, num_classes=opts.num_classes)


    model_dict = model.state_dict()
    weights_dict = torch.load(opts.ckpt, map_location=device)['model_state']
    for k, v in weights_dict.items():
        model_dict[k.replace('module.', '')] = weights_dict[k]
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    time_before_val = time.time()
    accuracy, error, false, score,score_gaus = validate(opts=opts, model=model, loader=val_loader, device=device,
                                                        metrics_seg=metrics_seg, metrics_gaus=metrics_gaus)

    time_after_val = time.time()
    print('Time_val = %f' % (time_after_val - time_before_val))
    print("Accuracy:" + str(accuracy))
    print("Forget:" + str(error))
    print("Alarms:" + str(false))

    print(metrics_seg.to_str(score))
    
    print(metrics_seg.to_str(score_gaus))

    

if __name__ == '__main__':
    main()
