"""
Parts of the code are adapted from 
https://github.com/microsoft/MeshGraphormer/blob/main/docs/EXP.md

"""


from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.metric_pampjpe import get_alignMesh


import argparse
import os

import os.path as op
import code
import json
import zipfile
import torch
import numpy as np


def load_pred_json(filepath):
    archive = zipfile.ZipFile(filepath, 'r')
    jsondata = archive.read('pred.json')
    reference = json.loads(jsondata.decode("utf-8"))
    return reference[0], reference[1]


def multiscale_fusion(output_dir):
    s = '10'
    filepath = output_dir + 'hamba-ckptxx-sc10_rot0-pred.zip'
    ref_joints, ref_vertices = load_pred_json(filepath)
    ref_joints_array = np.asarray(ref_joints)
    ref_vertices_array = np.asarray(ref_vertices)
    if len(ref_joints_array.shape) == 4 and ref_joints_array.shape[1] == 1:
        ref_joints_array = ref_joints_array[:, 0, :, :] # (3960, 1, 21, 3) --> (3960, 21, 3)
        ref_vertices_array = ref_vertices_array[:, 0, :, :]

    rotations = [0.0]
    for i in range(1,10):
        rotations.append(i*10)
        rotations.append(i*-10)
    
    scale = [0.7,0.8,0.9,1.0,1.1]
    multiscale_joints = []
    multiscale_vertices = []

    counter = 0
    for s in scale:
        for r in rotations:
            setting = 'sc%02d_rot%s'%(int(s*10),str(int(r)))
            filepath = output_dir+'hamba-ckptxx-'+setting+'-pred.zip'

            if not os.path.exists(filepath):
                print("Not Existing!!!! skip this file, need to double check :", filepath)
                continue

            joints, vertices = load_pred_json(filepath)
            joints_array = np.asarray(joints)
            vertices_array = np.asarray(vertices)

            if len(joints_array.shape) == 4 and ref_joints_array.shape[1] == 1:
                joints_array = joints_array[:, 0, :, :]         # (3960, 1, 21, 3) --> (3960, 21, 3)
                vertices_array = vertices_array[:, 0, :, :]
           
            pa_joint_error, pa_joint_array, _ = get_alignMesh(joints_array, ref_joints_array, reduction=None)
            pa_vertices_error, pa_vertices_array, _ = get_alignMesh(vertices_array, ref_vertices_array, reduction=None)
            
            print('--------------------------')
            print('scale:', s, 'rotate', r)
            print('PAMPJPE:', 1000*np.mean(pa_joint_error))
            print('PAMPVPE:', 1000*np.mean(pa_vertices_error))
            multiscale_joints.append(pa_joint_array)
            multiscale_vertices.append(pa_vertices_array)
            counter = counter + 1

    overall_joints_array = ref_joints_array.copy()
    overall_vertices_array = ref_vertices_array.copy()
    for i in range(counter):
        overall_joints_array += multiscale_joints[i]
        overall_vertices_array += multiscale_vertices[i]

    overall_joints_array /= (1+counter)
    overall_vertices_array /= (1+counter)
    pa_joint_error, pa_joint_array, _ = get_alignMesh(overall_joints_array, ref_joints_array, reduction=None)
    pa_vertices_error, pa_vertices_array, _ = get_alignMesh(overall_vertices_array, ref_vertices_array, reduction=None)
    print('--------------------------')
    print('overall:')
    print('PAMPJPE:', 1000*np.mean(pa_joint_error))
    print('PAMPVPE:', 1000*np.mean(pa_vertices_error))

    joint_output_save = overall_joints_array.tolist()
    mesh_output_save = overall_vertices_array.tolist()

    json_path = './pred.json'
    print(f'save results to {json_path}')
    with open(json_path, 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)

    filepath = output_dir + '/hamba-ckptxx-sc-rot-final-pred.zip'
    resolved_submit_cmd = 'zip ' + filepath + '  ' +  json_path
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = f'rm {json_path}'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    print(f'saved results to {filepath}')


def run_multiscale_inference(base_path, checkpoint, mode, output_dir):
    
    if mode==True:
        rotations = [0.0]
        for i in range(1,10):
            rotations.append(i*10)
            rotations.append(i*-10)
        scale = [0.7,0.8,0.9,1.0,1.1]
    else:
        rotations = [0.0]
        scale = [1.0] 

    job_cmd = "python eval/test_on_augmentation/run_hand_wScale_wRot.py " \
        "--base_path %s " \
        "--checkpoint %s " \
        "--rot %f " \
        "--sc %s " \
        "--out %s"

    for s in scale:
        for r in rotations:
            # hamba-ckptxx-sc10_rot-80-pred.zip
            # hamba-ckptxx-sc10_rot80-pred.zip
            setting = 'sc%02d_rot%s'%(int(s*10),str(int(r)))
            filename = 'hamba-ckptxx-' + setting + '-pred.zip'
            curr_path = os.path.join(output_dir, filename)
            if os.path.exists(curr_path):
                print(f"This file has been already processed!!! {curr_path}")
                continue
            else:
                print(f"Processing... {curr_path}")
                resolved_submit_cmd = job_cmd%(base_path, checkpoint, r, s, output_dir)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)

def main(args):
    base_path = args.base_path
    checkpoint = args.checkpoint
    mode = args.multiscale_inference
    output_dir = args.output_dir
    
    run_multiscale_inference(base_path, checkpoint, mode, output_dir)

    if mode==True:
        multiscale_fusion(output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint in the folder")
    parser.add_argument('--base_path', type=str, default='~/dataset/hand_vim_eval_data',
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--checkpoint', type=str, 
                    default='logs/train/runs/hamba.ckpt')
    parser.add_argument("--multiscale_inference", default=False, action='store_true',) 
    parser.add_argument("--output_dir", 
                        default='eval/freihand/results_multi_fusion/', 
                        type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    args = parser.parse_args()
    main(args)
