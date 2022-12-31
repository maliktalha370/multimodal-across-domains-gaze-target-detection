import multiprocessing
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from joblib import Parallel, delayed
from skimage.transform import resize
# from cv2 import resize  
from timm.utils import AverageMeter
from torchvision.transforms import transforms
import cv2
import wandb
from os.path import join
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


    # Get loss functions
    # mse_loss = nn.MSELoss(reduction="none")
    # adv_loss = nn.NLLLoss()
    # multimodal_loss = nn.MSELoss(reduction="none")

    # Get optimizer
    optimizer = get_optimizer(model, lr=config.lr)
    optimizer.zero_grad()

    # Do an evaluation or continue and prepare training
    # if config.eval_weights:
    print("Preparing evaluation")

    pretrained_dict = torch.load(config.eval_weights, map_location=device)
    pretrained_dict = pretrained_dict.get("model_state_dict") or pretrained_dict.get("model")

    model = load_pretrained(model, pretrained_dict)

    evaluate(config, model, device, target_test_loader)



def evaluate(config, model, device, loader):
    model.eval()

    output_size = config.output_size
    print_every = config.print_every



    with torch.no_grad():
        for batch, data in enumerate(loader):
            (
                path,
                images,
                depths,
                faces,
                head_channels,
                _,
                eye_coords,
                gaze_coords,
                _,
                img_size,
                head_bbox
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
                delayed(evaluate_one_item)(path[b_i], batch, b_i, images[b_i].cpu().numpy(),
                    gaze_heatmap_pred[b_i],output_size,config,
                    [head_bbox[0][b_i], head_bbox[1][b_i], head_bbox[2][b_i], head_bbox[3][b_i]]
                )
                for b_i in range(len(gaze_coords))
            )
            # for b_i in range(len(gaze_coords)):
            #     evaluate_one_item(batch, b_i, images[b_i].cpu().numpy(),
            #     gaze_heatmap_pred[b_i],output_size
            # )
            # print(metrics)
#             for metric in metrics:
#                 if metric is None:
#                     continue

#                 auc_score, min_dist, avg_dist, min_ang_err, avg_ang_err = metric

#                 auc_meter.update(auc_score)
#                 min_dist_meter.update(min_dist)
#                 min_ang_error_meter.update(min_ang_err)
#                 avg_dist_meter.update(avg_dist)
#                 avg_ang_error_meter.update(avg_ang_err)

#             if (batch + 1) % print_every == 0 or (batch + 1) == len(loader):
#                 print(
#                     f"Evaluation - BATCH {(batch + 1):04d}/{len(loader)} "
#                     f"\t AUC {auc_meter.avg:.3f}"
#                     f"\t AVG. DIST. {avg_dist_meter.avg:.3f}"
#                     f"\t MIN. DIST. {min_dist_meter.avg:.3f}"
#                     f"\t AVG. ANG. ERR. {avg_ang_error_meter.avg:.3f}"
#                     f"\t MIN. ANG. ERR. {min_ang_error_meter.avg:.3f}"
#                 )

#     return (
#         auc_meter.avg,
#         min_dist_meter.avg,
#         avg_dist_meter.avg,
#         min_ang_error_meter.avg,
#         avg_ang_error_meter.avg,
#     )


def evaluate_one_item(path, 
                      batch_no,
                      b_i, img,
    gaze_heatmap_pred,
    output_size,
    config,
    head_bbox,
                      
):
    print('path in 1' , path)
    new_img = cv2.imread(join(config.target_dataset_dir, path))
    # Min distance: minimum among all possible pairs of <ground truth point, predicted point>
    pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred)
    norm_p = torch.tensor([pred_x / float(output_size), pred_y / float(output_size)])
    norm_p_np = norm_p.numpy()
    converted = []
    for data in head_bbox:
        converted.append(int(data.numpy()))
    dst = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(dst, (converted[0], converted[1]),
                  (converted[2], converted[3]),
                  (255, 0, 0), 2)
    cv2.circle(dst, (int(norm_p_np[0] * new_img.shape[1]), int(norm_p_np[1] * new_img.shape[0])),
               int(new_img.shape[1] / 50.0), (255, 0, 0), -1)

    # plt.imshow(dst, cmap='gray')
    # plt.show()

    cv2.imwrite('output/' + str(batch_no) + '_' + str(b_i) + '.jpg', dst)
    # print(np.multiply(norm_p.numpy(), img_shape[1:]))
    
    # print('New Iter' , norm_p)
    return np.multiply(norm_p.numpy(), img.shape[1:])
if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    load_dotenv()

    main(get_config())
