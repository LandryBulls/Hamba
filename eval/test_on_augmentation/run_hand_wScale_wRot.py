"""
Parts of the code are adapted from 
https://github.com/lmb-freiburg/freihand/blob/master/pred.py

"""


from __future__ import print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from freihand.utils.fh_utils import *
from utils.image_ops import rgb_processing


import argparse
from tqdm import tqdm
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



def main(args, model, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """

    base_path, pred_out_path = args.base_path, args.out
    if not os.path.exists(pred_out_path):
        os.makedirs(pred_out_path)

    #### augm paprams start ####
    rot, sc = args.rot, args.sc
    flip = 0            # flipping
    pn = np.ones(3)     # per channel pixel-noise
    print("!!!>>> rot, sc: ", rot, sc)
    #### augm paprams end ####

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
        rgb_img = read_img(idx, base_path, set_name)
        
        ## do augmentation
        scale = 1.0
        h, w, _ = rgb_img.shape
        img_size = (h, w)
        center = (h * 0.5, w * 0.5)

        if rot > 0.0 or rot < 0.0 or sc > 1.0 or sc < 1.0:
            # cv2.imwrite("demo_out/before_aug.jpg", rgb_img[:, :, ::-1])
            # print("center, sc*scale, rot, flip, pn, img_size: ", center, sc*scale, rot, flip, pn, img_size)
            rgb_img = rgb_processing(rgb_img, center, sc*scale, rot, flip, pn, img_size)
            # cv2.imwrite(f"demo_out/after_aug_{sc}.jpg", rgb_img[:, :, ::-1])
        
        # use some algorithm for prediction
        xyz, verts = get_xyz_verts(
            rgb_img,
            model,
            np.array(K_list[idx]),
            scale_list[idx],
            xyz_list[idx]
        )
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]
    print('save results to pred.json')
    output_json_file = 'pred.json'
    print('save results to ', output_json_file)
    with open(output_json_file, 'w') as f:
        json.dump([xyz_pred_list, verts_pred_list], f)

    inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
    output_zip_file = pred_out_path + '/hamba-ckptxx' + '-' + inference_setting +'-pred.zip'

    resolved_submit_cmd = 'zip ' + output_zip_file + ' ' + output_json_file
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm %s'%(output_json_file)
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def get_xyz_verts(img, model, K, scale, gt_keypoints_3d):
    """ Predict joints and vertices from a given sample.
        img: (224, 224, 3) RGB image.
        K: (3, 3) camera intrinsic matrix.
        scale: () scalar metric length of the reference bone,
                  which was calculated as np.linalg.norm(xyz[9] - xyz[10], 2),
                  i.e. it is the length of the proximal phalangal bone of the middle finger.
    """
    # IMG_SIZE = 224
    IMG_SIZE = 256

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
    cx = cx / 224 * IMG_SIZE
    cy = cy / 224 * IMG_SIZE
    focal_length = np.array([fx, fy])
    camera_center = np.array([cx, cy])
    gt_cam_t = torch.zeros(1, 3)
    gt_keypoints_2d = perspective_projection((torch.from_numpy(np.array(gt_keypoints_3d))).unsqueeze(0), 
                                             gt_cam_t,
                                             (torch.from_numpy(focal_length / IMG_SIZE).unsqueeze(0)), 
                                             (torch.from_numpy(camera_center / IMG_SIZE).unsqueeze(0)))
         
    gt_keypoints_2d = gt_keypoints_2d - 0.5
    gt_keypoints_2d = torch.cat((gt_keypoints_2d, torch.ones_like(gt_keypoints_2d[..., :1])), dim=-1).float()

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

    with torch.no_grad():
        out = model(batch)
    xyz   = out['pred_keypoints_3d']    # 3D coordinates of the 21 joints
    verts = out['pred_vertices']        # 3D coordinates of the shape vertices

    if len(xyz.shape) == 3 and xyz.shape[0] == 1:
        xyz = xyz[0]
        verts = verts[0]

    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--base_path', type=str, default='~/datasets/hand_vim_eval_data', 
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--checkpoint', type=str, 
                        default='logs/train/runs/last.ckpt')
    parser.add_argument('--out', type=str, 
                        default='./eval/freihand/')

    parser.add_argument("--rot", default=0, type=float) 
    parser.add_argument("--sc", default=1.0, type=float) 
    args = parser.parse_args()
    
    model, model_cfg = load_hamba(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    main(
        args,
        model,
        set_name='evaluation',
    )
