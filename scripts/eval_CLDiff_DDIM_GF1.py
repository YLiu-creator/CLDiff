# import os
# import torch
# # select the GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, "1"))

import os
import argparse
import sys
import random
from metrics import StreamSegMetrics
sys.path.append(".")
import numpy as np
import torch
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


from guided_diffusion import dist_util, logger
from datasets.gf1_dataset import GF1_FULL_SUP_condFEATS
import multiprocessing

from collections import OrderedDict

seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def create_argparser():
    defaults = dict(
        data_name = 'GF1',
        data_dir="../dataset/GF1_datasets/datasets_321/data",

        model_version='swinB_MFR_diffDec_2stage',

        clip_denoised=True,
        num_samples=1,
        batch_size=8,
        use_ddim=True,
        dpm_solver=False,

        model_path="../checkpoints/swinB_MFR_diffDec_2stage/best.pt",         #path to pretrain model
        num_ensemble=1,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='../results/CLDiff_oup',
        tar_dir='../dataset/GF1_datasets/datasets_321/data/SegmentationClass/',
        multi_gpu = False, #"0,1,2"
        debug = True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)

    os.makedirs(args.out_dir,exist_ok=True)
    logger.configure(dir = args.out_dir)

    args.model_version = 'swinB_MFR_Gauss_Diff_2stage'
    args.diffusion_steps=20
    args.use_ddim=True

    if args.data_name == 'GF1':
        print('GF1 image_size set to: ' + str((args.image_size, args.image_size)))
        train_dst = GF1_FULL_SUP_condFEATS(root=args.data_dir, image_set='trainval')
        args.in_ch = 5

    datal = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=False)
    data = iter(datal)
    print("The number of the GF-1 dataset samples:" + str(len(data)))

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())

    print("Model restored from %s" % args.model_path)

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()
    metrics = StreamSegMetrics(2)
    metrics.reset()
    from tqdm import tqdm
    for _ in tqdm(range(len(data))):

        b, cls_tar, cond, name, HOT, en_feat4x, en_feat8x, en_feat16x = next(data)

        c = torch.randn_like(b[:, :1, ...])
        img = torch.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'GF1':
            slice_ID=name[0].split("/")[-1].split('.')[0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()

            if not args.use_ddim:
                print('No using DDIM')
                sample_fn = diffusion.p_sample_loop_known

                sample, x_noisy, org, cal, cal_out, diff_oup = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img,
                    en_feat4x, en_feat8x, en_feat16x,
                    step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            else:
                print('Test using DDIM')
                sample_fn = diffusion.ddim_sample_loop_known

                sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img,
                    name,
                    en_feat4x, en_feat8x, en_feat16x,
                    step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

            end.record()
            torch.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            cal_out = cal_out.detach().cpu().numpy()

            for b_th in range(args.batch_size):
                save_name = name[b_th].split('/')[-1].replace('.npy','')

                if i == 0:
                    pred = cal_out[b_th, ...].squeeze()
                    output = np.zeros((pred.shape[0], pred.shape[1]), dtype=int)
                    output[pred >= 0.3] = 1
                    target = np.load(args.tar_dir + save_name+'.npy')[:-1, :-1]
                    metrics.update(target, output)


    score = metrics.get_results()
    print(metrics.to_str(score))





if __name__ == "__main__":

    main()
