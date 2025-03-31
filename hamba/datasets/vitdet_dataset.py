from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

from hamba.datasets.utils import fliplr_keypoints, trans_point2d
from hamba.datasets.image_dataset import FLIP_KEYPOINT_PERMUTATION

import mediapipe as mp
mp_hands = mp.solutions.hands


DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 keypoints_2d_arr=None,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)
        self.keypoints_2d_arr = keypoints_2d_arr.astype(np.float32)

        self.mp_hand = mp_hands.Hands(
            # static_image_mode=True,
            # model_complexity=1, # default 1
            # max_num_hands=1, # default 2
            # min_detection_confidence=0.5, 
            # min_tracking_confidence=0.5,
            min_detection_confidence=0.2, 
            min_tracking_confidence=0.2
        )

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None) # model_cfg.MODEL.BBOX_SHAPE = [192,256]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        right = self.right[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        img_height, img_width, img_channels = cvimg.shape
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)

        # output_path = "./demo_out/before_patch_vis.jpg"
        # cv2.imwrite(output_path, cvimg)
        # print("cvimg.shape: ", cvimg.shape)
        # print("output_path: ", output_path)
        
        print("cvimg.shape: ", cvimg.shape) # cvimg.shape:  (1600, 1182, 3)
        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        # process keypoint_2d
        if self.keypoints_2d_arr is None:
            keypoints_2d = None
        else:
            keypoints_2d = self.keypoints_2d_arr[idx]
            # bbox = self.boxes[idx]
            # curr_bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            print("keypoints_2d.shape: ", keypoints_2d.shape)
            # print("keypoints_2d: ", keypoints_2d)
            # keypoints_2d[:, 0] -= (center_x - curr_bbox_size/2)
            # keypoints_2d[:, 1] -= (center_y - curr_bbox_size/2)
            # keypoints_2d[:, 0] -= (center_x - bbox_size/2)
            # keypoints_2d[:, 1] -= (center_y - bbox_size/2)
            if flip:
                keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, FLIP_KEYPOINT_PERMUTATION)

            for n_jt in range(len(keypoints_2d)):
                keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
            keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5
            
            ## visual
            # un-normalization
            img_patch_rgb = np.copy(img_patch)
            for n_c in range(min(self.img_cv2.shape[2], 3)):
                img_patch_rgb[n_c, :, :] = img_patch_rgb[n_c, :, :] * self.std[n_c] + self.mean[n_c]
            img_patch_rgb = img_patch_rgb.transpose(1, 2, 0)
            
            ############################
            ## keypoints_2d from mediapipe start
            results = self.mp_hand.process(img_patch_rgb.astype(np.uint8))
            print('Handedness:', results.multi_handedness) # left or right
            hand_landmarks = results.multi_hand_landmarks
            if hand_landmarks is None:
                item = {
                    'img': img_patch,
                    'personid': int(self.personid[idx]),
                    'is_valid': -1,
                    'keypoints_2d': np.zeros((21, 3)).astype(np.float32),
                    'box_center': np.array([0,0]).astype(np.float32),
                    'img_size': np.array([0,0]).astype(np.float32),
                    'box_size': 0,
                    'right': 1
                    }
                return item
            
            hand_landmark_obj = hand_landmarks[0].landmark
            keypoints_2d_list = []
            for point in hand_landmark_obj:
                x = point.x * patch_width
                y = point.y * patch_width
                # z = point.visibility
                z = 1.0
                keypoints_2d_list.append([x, y, z])
            keypoints_2d_224 = np.asarray(keypoints_2d_list).astype(np.float32)
            
            # image had been already flip
            # if flip:
            #     keypoints_2d_224 = fliplr_keypoints(keypoints_2d_224, patch_width, FLIP_KEYPOINT_PERMUTATION)

            keypoints_2d_224[:, :-1] = keypoints_2d_224[:, :-1] / patch_width - 0.5
            keypoints_2d = keypoints_2d_224
            ## keypoints_2d from mediapipe end
            ############################

            # keypoints_2d_vis = (keypoints_2d[:, :2] + 0.5) * patch_width 
            # for point in keypoints_2d_vis[:, :2]:
            #     x, y = map(int, point)
            #     cv2.circle(img_patch_rgb, (x, y), 5, (255, 0, 0), -1) 
            # output_path = "./demo_out/joint_vis.jpg"
            # cv2.imwrite(output_path, img_patch_rgb[:, :, ::-1])
            # print("img_patch_rgb.shape: ", img_patch_rgb.shape)
            # print("output_path: ", output_path)

            conf_threshold = 0.5
            invalid_mask = np.logical_or(np.abs(keypoints_2d[:, 0]) > conf_threshold, np.abs(keypoints_2d[:, 1]) > conf_threshold)
            keypoints_2d[invalid_mask, 2] = 0
            # valid_index = keypoints_2d[:, -1] < 0.3
            # print("norm keypoints_2d: ", keypoints_2d)

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = self.right[idx].copy()
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['is_valid'] = 1

        # print("item['box_center'].shape: ", item['box_center'].shape)
        # print("item['box_size'].shape: ", item['box_size'].shape)
        # print("item['img_size'].shape: ", item['img_size'].shape)
        # print("keypoints_2d.shape: ", keypoints_2d.shape)
        # print("item['right'].shape: ", item['right'].shape)
        # print("-1")

        return item
