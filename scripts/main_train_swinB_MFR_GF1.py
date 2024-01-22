import os
import random
import argparse
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm
from torch.utils import data

from datasets.gf1_dataset import GF1_WEAK_SUP, GF1_FULL_SUP
from guided_diffusion.mfrLoss import SSIM
from metrics import StreamSegMetrics
from guided_diffusion.swinB_MFR_parall import swin_MFR_Parall
from guided_diffusion.swinB_MFR_local import swin_MFR_Local
from guided_diffusion.swinB_MFR_regional import swin_RAM_Regional


def get_argparser():
    parser = argparse.ArgumentParser()

    # Save position
    parser.add_argument("--save_dir", type=str, default='../checkpoints/ckpt_gf1/',
                        help="path to Dataset")

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='..dataset/GF1_datasets/datasets_321/data/',
                        help="path to Dataset")

    # Model Options
    parser.add_argument("--model", type=str, default='swinB_MFR_GF1',
                        choices=['swinB_MFR_GF1', 'swinB_regionMFR_GF1', 'swinB_localMFR_GF1'],
                        help='model name')

    parser.add_argument("--num_classes", type=int, default=2, help="num classes (default: None)")
    parser.add_argument("--gpu_id", type=str, default='2', help="GPU ID")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=300000,
                        help="epoch number (default: 100k)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.0001)")

    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--optimizer_strtegy", type=str, default='SGD', choices=['AdamW', 'SGD'],
                        help="Optimizer strtegies")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step', 'checkUpdate'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--loss_type", type=str, default='hybrid',
                        choices=['cross_entropy', 'focal_loss'],
                        help="loss type (default: False)")

    parser.add_argument("--ckpt", default=None,
                        help="restore from checkpoint")
    parser.add_argument('--freeze-layers', type=bool, default=False)

    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval of loss (default: 10)")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'gf1':
        train_dst = GF1_WEAK_SUP(root=opts.data_root, image_set='train')
        val_dst = GF1_WEAK_SUP(root=opts.data_root, image_set='trainval')
    return train_dst, val_dst


def validate(opts, model, loader, device):
    """Do validation and return specified samples"""
    right, forgot, alarm, all = 0, 0, 0, 0

    with torch.no_grad():
        for i, (images, block_label, HOT, name) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = block_label.to(device, dtype=torch.long)

            img_cls, seg_pre, cond_feat = model(images)

            preds = img_cls.detach().max(dim=1)[1].cpu().numpy()
            labels = labels.cpu().numpy()

            for b in range(seg_pre.shape[0]):
                if preds[b] == labels[b]:
                    right += 1
                    all += 1
                elif preds[b] < labels[b]:
                    forgot += 1
                    all += 1
                elif preds[b] > labels[b]:
                    alarm += 1
                    all += 1

        accuracy = right / all
        error = forgot / all
        false = alarm / all

    return accuracy, error, false


def main():
    opts = get_argparser().parse_args()

    save_dir = os.path.join(opts.save_dir + opts.model + '/')
    os.makedirs(save_dir, exist_ok=True)
    print('Save position is %s\n' % (save_dir))

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.batch_size,
                                   drop_last=True, pin_memory=False)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.batch_size,
                                 drop_last=True, pin_memory=False)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))


    val_interval = int(len(train_dst) / (opts.batch_size * 2))


    # model = swin_MFR_Region(in_chans=4, num_classes=opts.num_classes)
    # model = swin_MFR_Local(in_chans=4, num_classes=opts.num_classes)
    model = swin_MFR_Parall(in_chans=4, num_classes=opts.num_classes)
    model = nn.DataParallel(model)
    model.to(device)

    # Set up optimizer_strtegy
    if opts.optimizer_strtegy == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer_strtegy == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.5)

    # Set up criterion
    if opts.loss_type == 'hybrid':
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_epochs": cur_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_epoch_loss,
        }, path)
        print("Model saved as %s\n\n" % path)

    # ==========   Train Loop   ==========#

    iter_loss, iter_cls, iter_mse,iter_seg = 0, 0, 0,0
    train_loss = []
    no_optim = 0
    best_epoch = 1
    learning_rate = []
    train_accuracy = list()

    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        train_epoch_loss = 0
        for (images, block_label, HOT, name) in train_loader:
            if (cur_itrs) == 0 or (cur_itrs) % opts.print_interval == 0:
                t1 = time.time()

            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            block_label = np.squeeze(block_label)
            img_true = block_label.to(device, dtype=torch.long)
            HOT = HOT.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            cls_pred, seg_pre, cond_feat = model(images)

            loss_cls = criterion_ce(cls_pred, img_true)
            loss_mse = criterion_mse(seg_pre, HOT)
            loss_ssmi = SSIM(seg_pre, HOT)

            alpah = 0.4
            loss = alpah * loss_cls + (1 - alpah) * (loss_mse + loss_ssmi)

            loss.backward()
            optimizer.step()

            np_loss_cls = loss_cls.detach().cpu().numpy()
            np_loss_mse = loss_mse.detach().cpu().numpy()
            np_loss_seg = loss_mse.detach().cpu().numpy()
            np_loss = loss.detach().cpu().numpy()

            iter_loss += np_loss
            iter_cls += np_loss_cls
            iter_mse += np_loss_mse
            iter_seg +=np_loss_seg

            train_epoch_loss += np_loss

            if (cur_itrs) % opts.print_interval == 0:
                iter_loss = iter_loss / opts.print_interval
                iter_cls = iter_cls / opts.print_interval
                iter_mse = iter_mse / opts.print_interval
                iter_seg = iter_seg / opts.print_interval

                train_loss.append(iter_loss)

                t2 = time.time()
                lr_print = scheduler.get_lr()[0]
                if lr_print not in learning_rate:
                    learning_rate.append(lr_print)

                print("Epoch %d, Itrs %d/%d, Loss=%f (%f/%f/%f), Learning rate = %f, Time = %f" %
                      (cur_epochs, cur_itrs, opts.total_itrs,
                       iter_loss, iter_cls, iter_mse, iter_seg, lr_print, t2 - t1,))

                iter_loss = 0.0
                iter_cls = 0.0
                iter_mse = 0.0
                iter_seg = 0.0


            # save the ckpt file per 5000 itrs
            if (cur_itrs) % val_interval == 0:
                print("validation...")
                model.eval()

                time_before_val = time.time()
                accuracy, error, false = validate(opts=opts, model=model,loader=val_loader, device=device)

                time_after_val = time.time()
                print('Time_val = %f' % (time_after_val - time_before_val))
                print("Accuracy:" + str(accuracy))
                print("Forget:" + str(error))
                print("Alarms:" + str(false))


                train_accuracy.append(accuracy)
                if accuracy > best_score:  # save best model
                    best_score = accuracy
                    save_ckpt(save_dir + 'best_%s_%s_epochs%s.pth' % (opts.model, opts.dataset, str(cur_epochs)))
                    no_optim = 0
                    best_epoch = cur_epochs
                else:
                    no_optim += 1
                    save_ckpt(save_dir + 'latest_%s_%s_epochs%s.pth' % (opts.model, opts.dataset, str(cur_epochs)))

                model.train()

            if cur_itrs >= opts.total_itrs:
                print(cur_itrs)
                print(opts.total_itrs)
                return

        if no_optim >= 2:
            print('Update learning rate')
            scheduler.step()  # update

        if no_optim > 5:
            print('Early stop at %d epoch' % cur_epochs)
            print('Best epoch is %d' % best_epoch)
            break

    print('Best epoch is %d' % best_epoch)


if __name__ == '__main__':
    main()
