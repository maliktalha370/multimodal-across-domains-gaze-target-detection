from skimage.transform import resize
import multiprocessing
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from joblib import Parallel, delayed

from timm.utils import AverageMeter
from torchvision.transforms import transforms

import wandb

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    print("AMP is not available")

from config import get_config
from datasets import get_dataset
from models import get_model, load_pretrained
from optimizer import get_optimizer
from utils import (
    get_angular_error,
    get_auc,
    get_heatmap_peak_coords,
    get_l2_dist,
    get_memory_format,
    get_multi_hot_map,
)


def main(config):
    # Create output dir if training
    if not config.eval_weights:
        os.makedirs(config.output_dir, exist_ok=True)

    # Load train and validation datasets
    print("Loading dataset")
    source_loader, target_loader, target_test_loader = get_dataset(config)

    # Define device
    device = torch.device(config.device)
    print(f"Running on {device}")

    # Load model
    print("Loading model")
    model = get_model(config, device=device)
    print(model)

    # Get loss functions
    mse_loss = nn.MSELoss(reduction="none")
    adv_loss = nn.NLLLoss()
    multimodal_loss = nn.MSELoss(reduction="none")

    # Get optimizer
    optimizer = get_optimizer(model, lr=config.lr)
    optimizer.zero_grad()

    # Do an evaluation or continue and prepare training

    if True:
        print("Preparing training")

        # Select best kernel for convolutions
        torch.backends.cudnn.benchmark = True

        # Allows to resume a run from a given epoch
        next_epoch = 0

        # This value is filled by checkpoint if resuming
        # Once assigned, run_id equals the id of a wandb run
        run_id = None

        # The next_epoch check makes sure that we start with init_weights even when resume is set to True but no
        # checkpoints are found
        if config.init_weights and next_epoch == 0:
            print("Loading init weights")

            pretrained_dict = torch.load(config.init_weights, map_location=device)
            pretrained_dict = pretrained_dict.get("model_state_dict") or pretrained_dict.get("model")

            model = load_pretrained(model, pretrained_dict)

            del pretrained_dict

        # Initialize wandb
        if run_id is not None and config.wandb:
            print(f"Resuming wandb run with id {run_id}")

            wandb.init(id=run_id, resume="must")
        elif config.wandb:
            run_id = wandb.util.generate_id()
            print(f"Starting a new wandb run with id {run_id}")

            wandb.init(
                id=run_id,
                config=config,
                tags=["spatial_depth_late_fusion", config.source_dataset, config.target_dataset],
            )
            wandb.run.name = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{config.model_id}_{config.tag}'
        else:
            # We force the run_id to a random string
            run_id = "0118999881999111725 3"


        print(f"Training from epoch {next_epoch + 1} to {config.epochs}. {len(source_loader)} batches per epoch")
        for ep in range(next_epoch, config.epochs):
            start = datetime.now()

            train_one_epoch(
                config,
                ep,
                model,
                device,
                source_loader,
                target_loader,
                mse_loss,
                adv_loss,
                multimodal_loss,
                optimizer,
                task_loss_amp_factor=config.task_loss_amp_factor,
                rgb_depth_source_loss_amp_factor=config.rgb_depth_source_loss_amp_factor,
                rgb_depth_target_loss_amp_factor=config.rgb_depth_target_loss_amp_factor,
                adv_loss_amp_factor=config.adv_loss_amp_factor,
            )

            print(f"Epoch {ep + 1} took {datetime.now() - start}")

            checkpoint = {
                "run_id": run_id,
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mse_loss": mse_loss,
                "adv_loss": adv_loss,
                "multimodal_loss": multimodal_loss,
            }

            # Save the model
            # We want to save the checkpoint of the last epoch, so that we can resume training later
            save_path = os.path.join(config.output_dir, "ckpt_last.pth")

            # Keep previous checkpoint until we are sure that this saving goes through successfully.
            backup_path = os.path.join(config.output_dir, "ckpt_last.backup.pth")
            if os.path.exists(save_path):
                os.rename(save_path, backup_path)

            # Try to save and load the latest checkpoint. If no exception, delete backup file. Otherwise, stop.
            try:
                torch.save(checkpoint, save_path)
                _ = torch.load(save_path, map_location=torch.device("cpu"))

                print(f"Checkpoint saved at {save_path}")
            except Exception as e:
                print(e)
                print("Unable to save or verify last checkpoint. Restoring previous checkpoint.")

                os.remove(save_path)
                os.rename(backup_path, save_path)

                exit(1)

            # Remove backup file
            if os.path.exists(backup_path):
                os.remove(backup_path)

            if config.save and ((ep + 1) % config.save_every == 0 or (ep + 1) == config.epochs):
                save_path = os.path.join(config.output_dir, f"ckpt_epoch_{ep + 1}.pth")
                torch.save(checkpoint, save_path)

                print(f"Checkpoint saved at {save_path}")

            if (ep + 1) % config.evaluate_every == 0 or (ep + 1) == config.epochs:
                print("Starting evaluation")
                auc, min_dist, avg_dist, min_ang_err, avg_ang_err = evaluate(config, model, device, target_test_loader)

                if config.wandb:
                    wandb.log(
                        {
                            "epoch": ep + 1,
                            "val/auc": auc,
                            "val/min_dist": min_dist,
                            "val/avg_dist": avg_dist,
                            "val/min_ang_err": min_ang_err,
                            "val/avg_ang_err": avg_ang_err,
                        }
                    )


def train_one_epoch(
    config,
    epoch,
    model,
    device,
    source_loader,
    target_loader,
    loss_rec,
    loss_domain,
    loss_multimodal,
    optimizer,
    task_loss_amp_factor=1,
    rgb_depth_source_loss_amp_factor=3,
    rgb_depth_target_loss_amp_factor=10,
    adv_loss_amp_factor=1,
):
    model.train()

    print_every = config.print_every

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    n_iter = len(source_loader)
    for batch in range(n_iter):
        data_source = next(source_iter)
        (
            s_rgb,
            s_depth,
            s_heads,
            s_masks,
            s_gaze_heatmaps,
            _,
            _,
            s_gaze_inside,
            _,
            _,
        ) = data_source
        for module in model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
        batch_size = s_rgb.shape[0]
        s_label = torch.zeros(batch_size, device=device).long()

        s_rgb = s_rgb.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_depth = s_depth.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_heads = s_heads.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_masks = s_masks.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_gaze_heatmaps = s_gaze_heatmaps.to(device, non_blocking=True)
        s_gaze_inside = s_gaze_inside.to(device, non_blocking=True).float()

        p = float(batch_size + epoch * n_iter) / config.epochs / n_iter
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        s_gaze_heatmap_pred, s_label_pred, s_rgb_rec, s_depth_rec = model(s_rgb, s_depth, s_masks, s_heads, alpha=alpha)

        s_gaze_heatmap_pred = s_gaze_heatmap_pred.squeeze(1)

        # Loss
        # L2 loss computed only for inside case
        s_rec_loss = loss_rec(s_gaze_heatmap_pred, s_gaze_heatmaps) * task_loss_amp_factor
        s_rec_loss = torch.mean(s_rec_loss, dim=1)
        s_rec_loss = torch.mean(s_rec_loss, dim=1)
        # Zero out loss when it's out-of-frame gaze case
        s_rec_loss = torch.mul(s_rec_loss, s_gaze_inside.mean(axis=1))
        s_rec_loss = torch.sum(s_rec_loss) / torch.sum(s_gaze_inside.mean(axis=1))

        total_loss = s_rec_loss


        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % print_every == 0 or (batch + 1) == n_iter:
            log = f"Training - EPOCH {(epoch + 1):02d}/{config.epochs:02d} BATCH {(batch + 1):04d}/{n_iter} "
            log += f"\t TASK LOSS (L2) {s_rec_loss:.6f}"



            print(log)

        if config.wandb:
            log = {
                "epoch": epoch + 1,
                "train/batch": batch,
                "train/task_loss": s_rec_loss,
                "train/loss": total_loss,
            }

            wandb.log(log)


def evaluate(config, model, device, loader):
    model.eval()

    output_size = config.output_size
    print_every = config.print_every

    auc_meter = AverageMeter()
    min_dist_meter = AverageMeter()
    avg_dist_meter = AverageMeter()
    min_ang_error_meter = AverageMeter()
    avg_ang_error_meter = AverageMeter()

    with torch.no_grad():
        for batch, data in enumerate(loader):
            (
                images,
                depths,
                faces,
                head_channels,
                _,
                eye_coords,
                gaze_coords,
                _,
                img_size,
                _,
            ) = data

            images = images.to(device, non_blocking=True, memory_format=get_memory_format(config))
            depths = depths.to(device, non_blocking=True, memory_format=get_memory_format(config))
            faces = faces.to(device, non_blocking=True, memory_format=get_memory_format(config))
            head = head_channels.to(device, non_blocking=True, memory_format=get_memory_format(config))

            gaze_heatmap_pred, _, _, _ = model(images, depths, head, faces)

            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()

            # Sets the number of jobs according to batch size and cpu counts. In any case, no less than 1 and more than
            # 8 jobs are allocated.
            n_jobs = max(1, min(multiprocessing.cpu_count(), 8, config.batch_size))
            metrics = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_one_item)(
                    gaze_heatmap_pred[b_i], eye_coords[b_i], gaze_coords[b_i], img_size[b_i], output_size
                )
                for b_i in range(len(gaze_coords))
            )

            for metric in metrics:
                if metric is None:
                    continue

                auc_score, min_dist, avg_dist, min_ang_err, avg_ang_err = metric

                auc_meter.update(auc_score)
                min_dist_meter.update(min_dist)
                min_ang_error_meter.update(min_ang_err)
                avg_dist_meter.update(avg_dist)
                avg_ang_error_meter.update(avg_ang_err)

            if (batch + 1) % print_every == 0 or (batch + 1) == len(loader):
                print(
                    f"Evaluation - BATCH {(batch + 1):04d}/{len(loader)} "
                    f"\t AUC {auc_meter.avg:.3f}"
                    f"\t AVG. DIST. {avg_dist_meter.avg:.3f}"
                    f"\t MIN. DIST. {min_dist_meter.avg:.3f}"
                    f"\t AVG. ANG. ERR. {avg_ang_error_meter.avg:.3f}"
                    f"\t MIN. ANG. ERR. {min_ang_error_meter.avg:.3f}"
                )

    return (
        auc_meter.avg,
        min_dist_meter.avg,
        avg_dist_meter.avg,
        min_ang_error_meter.avg,
        avg_ang_error_meter.avg,
    )


def evaluate_one_item(
    gaze_heatmap_pred,
    eye_coords,
    gaze_coords,
    img_size,
    output_size,
):
    # Remove padding and recover valid ground truth points
    valid_gaze = gaze_coords[gaze_coords != -1].view(-1, 2)
    valid_eyes = eye_coords[eye_coords != -1].view(-1, 2)

    # Skip items that do not have valid gaze coords
    if len(valid_gaze) == 0:
        return

    # AUC: area under curve of ROC
    multi_hot = get_multi_hot_map(valid_gaze, img_size)
    scaled_heatmap = resize(gaze_heatmap_pred, (img_size[1], img_size[0]))
    auc_score = get_auc(scaled_heatmap, multi_hot)

    # Min distance: minimum among all possible pairs of <ground truth point, predicted point>
    pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred)
    norm_p = torch.tensor([pred_x / float(output_size), pred_y / float(output_size)])
    all_distances = []
    all_angular_errors = []
    for index, gt_gaze in enumerate(valid_gaze):
        all_distances.append(get_l2_dist(gt_gaze, norm_p))
        all_angular_errors.append(get_angular_error(gt_gaze - valid_eyes[index], norm_p - valid_eyes[index]))

    # Average distance: distance between the predicted point and human average point
    mean_gt_gaze = torch.mean(valid_gaze, 0)
    avg_distance = get_l2_dist(mean_gt_gaze, norm_p)

    return auc_score, min(all_distances), avg_distance, min(all_angular_errors), np.mean(all_angular_errors)


if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    load_dotenv()

    main(get_config())

