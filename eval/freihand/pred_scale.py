from __future__ import print_function, unicode_literals

import sys
sys.path.append("~/codes/hamba")
import argparse
from tqdm import tqdm
from utils.fh_utils import *
import os
import yaml
import hydra
import torch
from pathlib import Path
import numpy as np
from hamba.configs import dataset_config, CACHE_DIR_HAMBA
from hamba.models import HAMBA, load_hamba
from hamba.utils import recursive_to
from hamba.utils.geometry import aa_to_rotmat, perspective_projection
import cv2
from hamba.models import HAMBA, download_models, load_hamba, DEFAULT_CHECKPOINT

def move_palm_to_wrist(kp3d):
    palm = kp3d[0]
    middle_mcp = kp3d[3]
    wrist = 2 * palm - middle_mcp
    kp3d[0] = wrist

    return kp3d

def main(base_path, pred_out_path, model, init_regression_model=None, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    K_list = json_load(os.path.join(base_path, '%s_K.json' % set_name))
    scale_list = json_load(os.path.join(base_path, '%s_scale.json' % set_name))
    xyz_list = json_load(os.path.join(base_path, '%s_xyz.json' % set_name))   # load 3d keypoints

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        if idx >= db_size(set_name):
            break

        # load input image
        img = read_img(idx, base_path, set_name)

        # use some algorithm for prediction
        xyz, verts = get_xyz_verts(
            img,
            model,
            init_regression_model,
            np.array(K_list[idx]),
            scale_list[idx],
            xyz_list[idx]
        )
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def get_xyz_verts(img, model, init_regression_model, K, scale, gt_keypoints_3d):
    """ Predict joints and vertices from a given sample.
        img: (224, 224, 3) RGB image.
        K: (3, 3) camera intrinsic matrix.
        scale: () scalar metric length of the reference bone,
                  which was calculated as np.linalg.norm(xyz[9] - xyz[10], 2),
                  i.e. it is the length of the proximal phalangal bone of the middle finger.
    """
    
    img_vis_1 = np.copy(img)
    img_vis_2 = np.copy(img)

    mean = 255. * np.array([0.485, 0.456, 0.406])
    std = 255. * np.array([0.229, 0.224, 0.225])
    img_patch = np.transpose(img, (2, 0, 1)).astype(np.float32)
    for n_c in range(min(img_patch.shape[2], 3)):
        img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    
    img_patch_hamba = np.transpose(img_patch, (1, 2, 0)).astype(np.float32)
    img_patch_hamba = cv2.resize(img_patch_hamba, (256, 256))
    img_patch_hamba = np.transpose(img_patch_hamba, (2, 0, 1)).astype(np.float32)

    img_patch = img_patch[None, ...]
    img_patch_hamba = img_patch_hamba[None, ...]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    focal_length = np.array([fx, fy])
    camera_center = np.array([cx, cy])
    gt_cam_t = torch.zeros(1, 3)
    gt_keypoints_2d = perspective_projection((torch.from_numpy(np.array(gt_keypoints_3d))).unsqueeze(0), 
                                             gt_cam_t,
                                             (torch.from_numpy(focal_length / 224).unsqueeze(0)), 
                                             (torch.from_numpy(camera_center / 224).unsqueeze(0)))
         
    gt_keypoints_2d = gt_keypoints_2d - 0.5
    gt_keypoints_2d = torch.cat((gt_keypoints_2d, torch.ones_like(gt_keypoints_2d[..., :1])), dim=-1).float()

    # IMG_SIZE = 224
    IMG_SIZE = 256
    batch = {
        'img': torch.from_numpy(img_patch_hamba).float(), # the input img is RGB order, not BGR(cv2)
        'keypoints_2d': gt_keypoints_2d.float(),
        "box_center": torch.from_numpy(np.array([IMG_SIZE/2, IMG_SIZE/2]).astype(np.float32)[None, ...]),
        "box_size": torch.from_numpy(np.array([IMG_SIZE]).astype(np.float32)[None, ...]),
        "img_size": torch.from_numpy(np.array([IMG_SIZE, IMG_SIZE]).astype(np.float32)[None, ...]),
        'right': torch.from_numpy(np.array([1]).astype(np.float32)[None, ...]),
        'personid': torch.from_numpy(np.array([0]).astype(np.float32)[None, ...]),
        }
    batch = recursive_to(batch, device)
    
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    xyz   = out['pred_keypoints_3d']    # 3D coordinates of the 21 joints
    verts = out['pred_vertices']        # 3D coordinates of the shape vertices

    curr_scale = torch.norm(xyz[0, 9] - xyz[0, 10], p=2)
    xyz = xyz / curr_scale * scale

    gt_keypoints_2d_vis = gt_keypoints_2d[0, :, :2].detach().cpu().numpy()
    gt_keypoints_2d_vis = (gt_keypoints_2d_vis + 0.5) * 224
    for i in range(gt_keypoints_2d_vis.shape[0]):
        point = gt_keypoints_2d_vis[i]
        x, y = map(int, point[:2])
        cv2.circle(img_vis_1, (x, y), 3, (0, 0, 255), -1) 
        cv2.putText(img_vis_1, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    xyz_2d  = out['pred_keypoints_2d']    # 3D coordinates of the 21 joints   = out['pred_keypoints_2d']    # 3D coordinates of the 21 joints
    gt_keypoints_2d_vis = xyz_2d[0, :, :2].detach().cpu().numpy()
    gt_keypoints_2d_vis = (gt_keypoints_2d_vis + 0.5) * 224
    for i in range(gt_keypoints_2d_vis.shape[0]):
        point = gt_keypoints_2d_vis[i]
        x, y = map(int, point[:2])
        cv2.circle(img_vis_2, (x, y), 3, (255, 0, 0), -1) 
        cv2.putText(img_vis_2, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    output_path = "demo_out/test_joint.jpg"
    img_vis = cv2.vconcat([img_vis_1, img_vis_2])
    cv2.imwrite(output_path, img_vis[:, :, ::-1])
    print(output_path)
    print("-" * 50)
    
    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--base_path', type=str, default='~/datasets/hand_vim_eval_data', 
                        help='Path to where the FreiHAND dataset is located.')
    
    checkpoint = "logs/train/runs/hamba.ckpt"
    parser.add_argument('--out', type=str, default='./eval/freihand/results/hamba.json')

    args = parser.parse_args()
    
    model, model_cfg = load_hamba(checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    main(
        args.base_path,
        args.out,
        model,
        set_name='evaluation',
    )
