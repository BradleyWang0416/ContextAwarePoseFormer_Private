import os
import sys
import shutil
import argparse
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from tensorboardX import SummaryWriter
import yaml
from easydict import EasyDict as edict

from mvn.models.conpose_vqvae_byBrad import CAPF_VQVAE
from mvn.models.loss import MPJPE, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
from mvn.utils import misc
from mvn.utils.cfg import config, update_config, update_dir
from mvn.datasets import utils as dataset_utils

from mvn.datasets.human36m3D_video_byBradley import H36m3D_MultiFrame, data_prefetcher_H36m3D_MultiFrame

joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")    # 'experiments/human36m/human36m.yaml'
    parser.add_argument('--eval', action='store_true', help="Only evaluation if set")
    parser.add_argument('--eval_dataset', type=str, default='val', choices=['train','val'],
                        help="Split for evaluation")
    parser.add_argument("--local_rank", type=int, help="Local rank on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--sync_bn', action='store_true', help="Use PyTorch convert_syncbn_model")
    parser.add_argument("--logdir", type=str, default="logs/", help="Logs path")
    parser.add_argument("--azureroot", type=str, default="", help="Root path for code")
    parser.add_argument("--frame", type=int, default=1, help="Frame number to use")
    parser.add_argument("--backbone", type=str, default='hrnet_32', choices=['hrnet_32', 'hrnet_48', 'cpn'], help="2D pose backbone")
    
    args = parser.parse_args()
    update_config(args.config)
    update_dir(args.azureroot, args.logdir)
    config.model.backbone.type = args.backbone
    
    return args

def setup_human36m_dataloaders(config, is_train, distributed_train, rank=None, world_size=None):
    train_dataloader = None
    if is_train:
        train_dataset = H36m3D_MultiFrame(
            designated_split='train',
        )
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if distributed_train else None
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=config.train.shuffle and train_sampler is None,
            sampler=train_sampler,
            num_workers=config.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    val_dataset = H36m3D_MultiFrame(
            designated_split='test',
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=config.val.shuffle,
        num_workers=config.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )
    return train_dataloader, val_dataloader, (train_sampler if is_train else None), None

def setup_dataloaders(config, is_train=True, distributed_train=False, rank=None, world_size=None):
    if config.dataset.kind == 'human36m':
        train_dl, val_dl, train_sampler, dist_size = setup_human36m_dataloaders(
            config, is_train, distributed_train, rank, world_size
        )
        # _, whole_val_dl, _, _ = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError(f"Unknown dataset: {config.dataset.kind}")
    return train_dl, val_dl, train_sampler, None, dist_size

def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"
    experiment_title = (config.title + "_" + model_name) if config.title else model_name
    experiment_title = prefix + "ConPose"  # Overwrite with "ConPose"
    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print(f"Experiment name: {experiment_name}")

    experiment_dir = os.path.join(config.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)

    # Copy the config file for reference
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer

def one_epoch_full(model, criterion, optimizer, scheduler, config, dataloader, device, epoch,
                   n_iters_total=0, is_train=True, lr=None, master=False,
                   whole_val_dataloader=None, dist_size=None):
    name = "train" if is_train else "val"
    if is_train:
        model.train()
        if config.model.backbone.fix_weights:
            if hasattr(model, 'module'):
                model.module.backbone.eval()
                model.module.tokenizer.train()
            else:
                model.backbone.eval()
                model.tokenizer.train()

    else:
        model.eval()

    epoch_loss_3d, N = 0.0, 0
    metric_dict = defaultdict(list)
    results = defaultdict(list)

    if is_train: num_batches_per_epoch = len(dataloader)

    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        # No forced iteration here; we let the training loop do it
        prefetcher_class = data_prefetcher_H36m3D_MultiFrame

        prefetcher = prefetcher_class(dataloader, device, is_train,
                                      config.val.flip_test,
                                      config.model.backbone.type)
        batch = prefetcher.next()
        data_len = len(dataloader)
        pbar = tqdm(total=data_len, desc=f"[{name} epoch {epoch+1}]", mininterval=10, disable=(not master or not sys.stdout.isatty()))

        num_iter = 0
        while batch is not None:
            
            if is_train:
                current_iter = epoch * num_batches_per_epoch + num_iter
                adjust_learning_rate(optimizer, current_iter, lr, warmup_iters=500, warmup_ratio=0.001)


            video_rgb, joint3d_image_affined, joint3d_image, img_ori_hw, factor_2_5d = batch
            # [B,T,256,192,3], [B,T,17,3], [B,T,17,3]
            # Flip-test logic (if needed)
            batch_size, num_frames = joint3d_image.shape[:2]

            if (not is_train) and config.val.flip_test:
                raise NotImplementedError
                pred = model(images_batch[:, 0],
                             kpts_2d_cpn[:, 0],
                             kpts_2d_cpn_crop[:, 0].clone())
                pred_flip = model(images_batch[:, 1],
                                  kpts_2d_cpn[:, 1],
                                  kpts_2d_cpn_crop[:, 1].clone())
                pred_flip[..., 0] *= -1
                pred_flip[..., joints_left + joints_right, :] = pred_flip[..., joints_right + joints_left, :]

                # MODIFIED BY BRADLEY 250717
                # keypoints_3d_pred = torch.mean(torch.cat((pred, pred_flip), dim=1), dim=1, keepdim=True)
                keypoints_3d_pred = torch.mean(torch.stack((pred, pred_flip), dim=1), dim=1)
                del pred_flip
            else:
                losses, recoverd_joints, joint3d_image_affined_normed_flattened = model(video_rgb, joint3d_image_affined, joint3d_image, img_ori_hw, train=is_train)



            if is_train and 'debugpy' in sys.modules and len(metric_dict['total_loss']) > 5:
                break    # For debugpy: run only 5 iterations per epoch



            if is_train:
            # Compute loss
                loss = sum(losses.values())    # loss: a scalar tensor
                metric_dict['total_loss'].append(loss.item())

                epoch_loss_3d += joint3d_image.size(0) * loss.item()
                N += joint3d_image.size(0)
                optimizer.zero_grad()
                if not torch.isnan(loss):
                    loss.backward()
                    if config.loss.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.loss.grad_clip / config.train.volume_net_lr
                        )
                    optimizer.step()

                    scheduler.step()

                pbar.set_postfix(loss=loss.item())

            else:
                results['keypoints_gt'].append(joint3d_image_affined_normed_flattened.detach().reshape(batch_size, num_frames, -1, 3).cpu().numpy())
                results['keypoints_pred'].append(recoverd_joints.detach().reshape(batch_size, num_frames, -1, 3).cpu().numpy())
                results['2.5d_factor'].append(factor_2_5d.cpu().numpy())
                results['img_ori_hw'].append(img_ori_hw.cpu().numpy())


            if is_train and num_iter % 100 == 0 or num_iter == data_len:
                print(f"epoch {epoch+1} | iter {num_iter}/{data_len} | loss: {loss.item():.4f}")
                pbar.update(50)

            num_iter += 1

            batch = prefetcher.next()

        pbar.close()

    if is_train:
        return epoch_loss_3d / max(1, N)

    # Evaluate
    if dist_size is not None:
        for term in ['keypoints_gt', 'keypoints_3d']:
            results[term] = torch.cat(results[term])
            buffer = [
                torch.zeros(dist_size[-1], *results[term].shape[1:], device=device)
                for _ in range(len(dist_size))
            ]
            scatter_tensor = torch.zeros_like(buffer[0])
            scatter_tensor[:results[term].shape[0]] = results[term]
            torch.distributed.all_gather(buffer, scatter_tensor)
            results[term] = torch.cat([t[:n] for t, n in zip(buffer, dist_size)], dim=0)

    if master:
        if dist_size is None:
            print('Evaluating...')
            keypoints_gt = np.concatenate(results['keypoints_gt'])    # [N,T,17,3]
            keypoints_pred = np.concatenate(results['keypoints_pred'])
            factor_2_5d = np.concatenate(results['2.5d_factor'])
            image_ori_hw = np.concatenate(results['img_ori_hw'])

            image_processed_hw = prefetcher.processed_image_shape   # (192,256)
            image_w = image_processed_hw[0]
            image_h = image_processed_hw[1]

            keypoints_pred_denorm = keypoints_pred.copy()
            keypoints_gt_denorm = keypoints_gt.copy()

            keypoints_pred_denorm[..., :2] = (keypoints_pred_denorm[..., :2] + np.array([1, 1])) * np.array([image_w // 2, image_h // 2])
            keypoints_pred_denorm[..., 2] = keypoints_pred_denorm[..., 2] * image_ori_hw[..., 0:1] / 2

            keypoints_gt_denorm[..., :2] = (keypoints_gt_denorm[..., :2] + np.array([1, 1])) * np.array([image_w // 2, image_h // 2])
            keypoints_gt_denorm[..., 2] = keypoints_gt_denorm[..., 2] * image_ori_hw[..., 0:1] / 2

            mpjpe_pixel = np.linalg.norm(keypoints_pred_denorm - keypoints_gt_denorm, axis=-1).mean().item()

            keypoints_pred_denorm = keypoints_pred_denorm * factor_2_5d[..., None, None]    # 单位从像素变成毫米
            keypoints_gt_denorm = keypoints_gt_denorm * factor_2_5d[..., None, None]

            mpjpe_millimeter = np.linalg.norm(keypoints_pred_denorm - keypoints_gt_denorm, axis=-1).mean().item()

            res = {
                'MPJPE_pixel': mpjpe_pixel,
                'MPJPE_millimeter': mpjpe_millimeter,
            }
        else:
            raise NotImplementedError
            res = whole_val_dataloader.dataset.evaluate(results['keypoints_gt'],
                                                        results['keypoints_3d'],
                                                        None, config)
        return res

def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False
    # torch.cuda.set_device(args.local_rank)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert os.environ["MASTER_PORT"], "Set MASTER_PORT or use PyTorch launcher"
    assert os.environ["RANK"], "Use PyTorch launcher and specify RANK"
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return True

def match_name_keywords(n, name_keywords):
    return any(b in n for b in name_keywords)

def adjust_learning_rate(optimizer, current_iter, lr, warmup_iters, warmup_ratio):
    if current_iter < warmup_iters:
        warmup_lr = lr * (warmup_ratio + (1 - warmup_ratio) * current_iter / warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

def main(args):
    is_distributed = init_distributed(args)
    master = True
    rank, world_size = None, None
    if is_distributed and os.environ["RANK"]:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        master = (rank == 0)

    # device = torch.device(args.local_rank if is_distributed else 0)
    device = torch.device(int(os.environ["LOCAL_RANK"]) if is_distributed else 0)
    config.train.n_iters_per_epoch = None

    # Backbone-specific config
    if args.backbone == 'hrnet_32':
        config.model.poseformer.base_dim = 32
    elif args.backbone == 'hrnet_48':
        config.model.backbone.checkpoint = 'data/pretrained/coco/pose_hrnet_w48_256x192.pth'
        config.model.backbone.STAGE2.NUM_CHANNELS = [48, 96]
        config.model.backbone.STAGE3.NUM_CHANNELS = [48, 96, 192]
        config.model.backbone.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
        config.model.poseformer.base_dim = 48
    elif args.backbone == 'cpn':
        config.train.batch_size = 256
        config.model.backbone.checkpoint = 'data/pretrained/coco/CPN50_256x192.pth.tar'
        config.model.poseformer.base_dim = 256

    model = CAPF_VQVAE(config, device)

    experiment_dir, writer = (None, None)
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
        shutil.copy('mvn/models/conpose.py', experiment_dir)
        shutil.copy('mvn/models/pose_dformer.py', experiment_dir)
        shutil.copy('train.py', experiment_dir)

    print("args:", args)
    print("Number of GPUs:", torch.cuda.device_count())

    # Load pretrained backbone if requested
    if config.model.backbone.init_weights:
        if args.backbone in ['hrnet_32', 'hrnet_48']:
            ret = model.backbone.load_state_dict(
                torch.load(config.model.backbone.checkpoint, weights_only=True),
                strict=False
            )
        else:  # cpn
            st_dict = torch.load(config.model.backbone.checkpoint, map_location=device)['state_dict']
            for k in list(st_dict.keys()):
                st_dict[k.replace("module.", "")] = st_dict.pop(k)
            ret = model.backbone.load_state_dict(st_dict, strict=True)
        print(ret)
        print(f"Loaded {args.backbone} from {config.model.backbone.checkpoint}")

    # Load model checkpoint for evaluation
    if args.eval:
        ckpt_path = f'checkpoint/best_epoch_{args.backbone}.bin'
        checkpoint = torch.load(ckpt_path, weights_only=True)['model']
        for k in list(checkpoint.keys()):
            checkpoint[k.replace("module.", "")] = checkpoint.pop(k)
        ret = model.load_state_dict(checkpoint, strict=False)   # 'strict' changed to False [By Bradley 250717]
        print(ret)
        print(f"Loaded checkpoint from {ckpt_path}")

    # Optionally convert to SyncBatchNorm
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)

    # Choose loss
    criterion_class = {
        "MPJPE": MPJPE,
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.loss.criterion]
    if config.loss.criterion == "MSESmooth":
        criterion = criterion_class(config.loss.mse_smooth_threshold).to(device)
    else:
        criterion = criterion_class().to(device)

    # Optimizer setup
    # lr = config.train.volume_net_lr
    # lr_decay = config.train.volume_net_lr_decay
    # param_dicts = [
    #     {
    #         "params": [p for n, p in model.volume_net.named_parameters() if p.requires_grad],
    #         "lr": lr,
    #     },
    # ]
    # optimizer = None
    # if not args.eval:
    #     optimizer = optim.AdamW(param_dicts, weight_decay=0.1)


    # Data loaders (NO forced iteration over them, so it won't take extra minutes)
    if master:
        print("Loading data...")

    train_dl, val_dl, train_sampler, whole_val_dl, dist_size = setup_dataloaders(
        config, distributed_train=is_distributed, rank=rank, world_size=world_size
    )
    ######################################### PCT ##################################################################################
    lr = 1e-2
    betas = (0.9, 0.999)
    weight_decay = 0.15

    # 用 AdamW 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )

    # 余弦退火学习率调度参数（来自配置文件）
    total_epochs = config.train.n_epochs
    min_lr_ratio = 1e-5

    # 假设 train_loader 已定义
    num_batches_per_epoch = len(train_dl)
    total_iters = total_epochs * num_batches_per_epoch
    min_lr = lr * min_lr_ratio

    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iters,
        eta_min=min_lr
    )

    # 可选：warmup实现（手动）
    
    ######################################### PCT ##################################################################################


    # No loop over train_dl, val_dl here

    if master:
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameter count: {model_params}")
        model_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Untrainable parameter count: {model_params}")

    if is_distributed:
        # model = DistributedDataParallel(model, device_ids=[device], output_device=args.local_rank)
        model = DistributedDataParallel(model, device_ids=[device], output_device=int(os.environ["LOCAL_RANK"]))

    # ------------------------------------------------------------------------
    # Training or Evaluation
    # ------------------------------------------------------------------------
    if not args.eval:
        min_loss = float('inf')
        for epoch in range(config.train.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            start_time = time.time()
            train_loss = one_epoch_full(
                model, criterion, optimizer, scheduler, config, train_dl,
                device, epoch, lr=lr, is_train=True, master=master
            )
            val_res = one_epoch_full(
                model, criterion, optimizer, scheduler, config, val_dl,
                device, epoch, is_train=False, master=master,
                whole_val_dataloader=whole_val_dl, dist_size=dist_size
            )

            if master:
                # errors_p1 = [val_res[k]['MPJPE'] * 1000 for k in val_res]
                # errors_p2 = [val_res[k]['P_MPJPE'] * 1000 for k in val_res]
                # mean_p1 = round(np.mean(errors_p1), 1)
                # mean_p2 = round(np.mean(errors_p2), 1)
                # train_loss_mm = train_loss * 1000

                train_loss_mm = train_loss
                mean_p1 = val_res['MPJPE_millimeter']
                mean_p2 = val_res['MPJPE_pixel']

                print(
                    f"[Epoch {epoch+1}] time: {((time.time() - start_time)/60):.2f}m | "
                    f"lr: {lr:.6f} | train-loss: {train_loss_mm:.3f} | "
                    f"mpjpe_millimeter: {mean_p1:.1f} | mpjpe_pixel: {mean_p2:.1f}"
                )

                # Save best checkpoint
                if mean_p1 < min_loss:
                    min_loss = mean_p1
                    ckpt_path = os.path.join(experiment_dir, "checkpoints/best_epoch.bin")
                    print(f"  --> New best model! Saving to {ckpt_path}")
                    torch.save({
                        'epoch': epoch + 1,
                        'lr': lr,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, ckpt_path)

            # Decay LR
            # lr *= lr_decay
            # for pg in optimizer.param_groups:
            #     pg['lr'] = lr

    else:
        dataloader = train_dl if args.eval_dataset == 'train' else val_dl
        val_res = one_epoch_full(
            model, criterion, None, config, dataloader,
            device, 0, is_train=False, master=master,
            whole_val_dataloader=whole_val_dl, dist_size=dist_size
        )
        if master:
            errors_p1 = []
            errors_p2 = []
            errors_vel = []
            for k in val_res:
                p1_ = val_res[k]['MPJPE'] * 1000
                p2_ = val_res[k]['P_MPJPE'] * 1000
                vel_ = val_res[k]['MPJVE'] * 1000
                print(f"{k}: p1={p1_:.2f}, p2={p2_:.2f}, e_vel={vel_:.2f}")
                errors_p1.append(p1_)
                errors_p2.append(p2_)
                errors_vel.append(vel_)
            print(
                "avg p1:", round(np.mean(errors_p1), 1),
                "p2:", round(np.mean(errors_p2), 1),
                "MPJVE:", round(np.mean(errors_vel), 2)
            )
            print("Done.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
