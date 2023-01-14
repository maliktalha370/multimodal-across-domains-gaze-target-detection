import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    print("AMP is not available")

# from optimizer import get_optimizer
from torchvision.transforms import transforms



from config import get_config
from utils import (
    get_heatmap_peak_coords,
    get_memory_format,
    get_head_mask
)
from models import get_model, load_pretrained
from datasets.transforms.ToColorMap import ToColorMap

class Inference:
    def __init__(self):
        self.config = get_config()


        self.input_size = self.config.input_size
        self.output_size = self.config.output_size

        # self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((self.input_size, self.input_size)), transforms.ToTensor()]
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on {self.device}")

        # Load model
        print("Loading model")
        model = get_model(self.config, device=self.device)

        # # Get optimizer
        # optimizer = get_optimizer(model, lr=self.config.lr)
        # optimizer.zero_grad()

        # Do an evaluation or continue and prepare training
        # if config.eval_weights:
        print("Preparing evaluation")

        pretrained_dict = torch.load(self.config.eval_weights, map_location=self.device)
        pretrained_dict = pretrained_dict.get("model_state_dict") or pretrained_dict.get("model")

        self.model = load_pretrained(model, pretrained_dict)
        self.model.eval()
    def run_overVideo(self):
        self.csv_path = self.config.elm_head
        self.data_dir = self.config.elm_frame
        column_names = [
            "path",
            "left",
            "top",
            "right",
            "bottom"]


        df = pd.read_csv(self.csv_path, sep=",", names=column_names, usecols=column_names, index_col=False)
        for i in df.index:



            path = df.loc[i, 'path']
            x_min = df.loc[i,'left']
            y_min = df.loc[i,'top']
            x_max = df.loc[i,'right']
            y_max = df.loc[i,'bottom']

            img = Image.open(os.path.join(self.config.elm_frame, path))
            img = img.convert("RGB")
            img_cp = np.array(img.copy())
            width, height = img.size

            # Expand face bbox a bit
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

            x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

            head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

            # Crop the face
            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Load depth image
            depth_path = path.replace("images", "depth")
            depth = Image.open(os.path.join(self.data_dir, depth_path))
            depth = depth.convert("L")

            # Apply transformation to images...
            if self.image_transform is not None:
                img = self.image_transform(img)
                face = self.image_transform(face)

            # ... and depth
            if self.depth_transform is not None:
                depth = self.depth_transform(depth)

            img = img.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
            depth = depth.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
            face = face.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
            head = head.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)

            gaze_heatmap_pred, _, _, _ = self.model(img, depth, head, face)

            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()

            # gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()
            pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred[0])
            norm_p = torch.tensor([pred_x / float(self.config.output_size), pred_y / float(self.config.output_size)])
            norm_p_np = norm_p.numpy()
            converted = list(map(int, [x_min, y_min, x_max, y_max]))

            dst = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
            cv2.rectangle(dst, (converted[0], converted[1]),
                          (converted[2], converted[3]),
                          (255, 0, 0), 2)
            cv2.circle(dst, (int(norm_p_np[0] * img_cp.shape[1]), int(norm_p_np[1] * img_cp.shape[0])),
                       int(img_cp.shape[1] / 50.0), (255, 0, 0), -1)

            starting_point = ((converted[0] + converted[2]) // 2, (converted[1] + converted[3]) // 2)
            ending_point = (int(norm_p[0] * img_cp.shape[1]), int(norm_p[1] * img_cp.shape[0]))
            cv2.line(dst, starting_point, ending_point, (255, 0, 0), 5)

            screen = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            plt.imshow(screen, cmap='gray')
            plt.show()
            # cv2.imwrite('results/VideoAtten/' + str(batch_no) + '_' + str(b_i) + '.jpg', screen)

    def run_overSingleFrame(self, img, depth, head_bbox):

        x_min = head_bbox[0]
        y_min = head_bbox[1]
        x_max = head_bbox[2]
        y_max = head_bbox[3]

        img = img.convert("RGB")
        img_cp = np.array(img.copy())
        width, height = img.size

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Load depth image
        depth = depth.convert("L")

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        img = img.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        depth = depth.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        face = face.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        head = head.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)

        gaze_heatmap_pred, _, _, _ = self.model(img, depth, head, face)

        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()

        # gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()
        pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred[0])
        norm_p = torch.tensor([pred_x / float(self.config.output_size), pred_y / float(self.config.output_size)])

        converted = list(map(int, [x_min, y_min, x_max, y_max]))
        starting_point = ((converted[0] + converted[2]) // 2, (converted[1] + converted[3]) // 2)
        ending_point = (int(norm_p[0] * img_cp.shape[1]), int(norm_p[1] * img_cp.shape[0]))

        return starting_point, ending_point


if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    load_dotenv()
    obj = Inference()
    cnf = get_config()
    csv_path = cnf.elm_head
    data_dir = cnf.elm_frame
    column_names = [
        "path",
        "left",
        "top",
        "right",
        "bottom"]

    df = pd.read_csv(csv_path, sep=",", names=column_names, usecols=column_names, index_col=False)
    for i in df.index:
        path = df.loc[i, 'path']
        img = Image.open(os.path.join(cnf.elm_frame, path))
        depth_path = path.replace("images", "depth")
        depth = Image.open(os.path.join(data_dir, depth_path))
        head_bbox = [df.loc[i, 'left'], df.loc[i, 'top'], df.loc[i, 'right'],df.loc[i, 'bottom']]
        start_pnt, end_pnt = obj.run_overSingleFrame(img, depth, head_bbox)
        img_cp = np.array(img.copy())
        cv2.line(img_cp, start_pnt, end_pnt, (255, 0, 0), 10)
        cv2.circle(img_cp, end_pnt, 12, (255, 0, 0), -1)
        plt.imshow(img_cp, cmap='gray')
        plt.show()
    print('All images successfully completed !!!!!')