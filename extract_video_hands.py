"""
Video hand extraction using Hamba - renders 2D keypoints on video
"""
import os
# Set OpenGL platform BEFORE any imports
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from hamba.configs import get_config
from hamba.models.hamba import HAMBA
from hamba.utils import recursive_to
from hamba.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamba.utils.renderer import cam_crop_to_full

from vitpose_model import ViTPoseModel


# MANO hand skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Colors for each finger (BGR)
FINGER_COLORS = [
    (255, 128, 0),   # Thumb - orange
    (0, 255, 0),     # Index - green
    (0, 255, 255),   # Middle - yellow
    (0, 128, 255),   # Ring - orange-red
    (255, 0, 255),   # Pinky - magenta
]


def load_hamba_no_renderer(checkpoint_path):
    """Load Hamba model without initializing the renderer"""
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)
    
    # Override config values
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    
    # Remove pretrained weights config
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        if 'PRETRAINED_WEIGHTS_INIT_REGRESSION' in model_cfg.MODEL.keys():
            model_cfg.MODEL.pop('PRETRAINED_WEIGHTS_INIT_REGRESSION')
        model_cfg.freeze()
    
    print("Loading checkpoint:", checkpoint_path)
    
    # Create model without renderer
    model = HAMBA(cfg=model_cfg, init_renderer=False)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    return model, model_cfg


def draw_hand_2d(img, keypoints_2d, is_right=True, confidence_threshold=0.3):
    """
    Draw 2D hand keypoints and skeleton on image.
    keypoints_2d: (21, 3) array with [x, y, confidence] for each joint
    """
    h, w = img.shape[:2]
    
    # Base color: green for right hand, blue for left
    base_color = (0, 255, 0) if is_right else (255, 128, 0)
    
    # Draw connections with finger-specific colors
    for conn_idx, (start_idx, end_idx) in enumerate(HAND_CONNECTIONS):
        finger_idx = conn_idx // 4  # Each finger has 4 connections
        color = FINGER_COLORS[finger_idx]
        
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            pt1 = keypoints_2d[start_idx]
            pt2 = keypoints_2d[end_idx]
            
            # Check confidence
            if len(pt1) > 2 and len(pt2) > 2:
                if pt1[2] < confidence_threshold or pt2[2] < confidence_threshold:
                    continue
            
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)  # Thicker lines
    
    # Draw joints
    for idx, pt in enumerate(keypoints_2d):
        if len(pt) > 2 and pt[2] < confidence_threshold:
            continue
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            # Wrist larger, fingertips medium - all bigger
            radius = 8 if idx == 0 else (7 if idx in [4, 8, 12, 16, 20] else 5)
            cv2.circle(img, (x, y), radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img


def project_3d_to_2d(joints_3d, cam_t, focal_length, img_center):
    """Project 3D joints to 2D image coordinates"""
    # Add camera translation
    points_cam = joints_3d + cam_t
    
    # Perspective projection
    x = points_cam[:, 0] / points_cam[:, 2] * focal_length + img_center[0]
    y = points_cam[:, 1] / points_cam[:, 2] * focal_length + img_center[1]
    
    # Add confidence of 1.0 for all projected points
    conf = np.ones(len(x))
    
    return np.stack([x, y, conf], axis=-1)


def process_video(args):
    """Process video and extract hand poses frame by frame"""
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load detector
    print("Loading detector...")
    from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamba
    cfg_path = Path(hamba.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.05  # Lower threshold
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # Keypoint detector
    print("Loading ViTPose...")
    cpm = ViTPoseModel(device)
    
    # Open video
    print(f"Opening video: {args.input_video}")
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Create output video writer
    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=min(total_frames, args.max_frames) if args.max_frames else total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.max_frames and frame_idx >= args.max_frames:
            break
        
        # Process every nth frame only
        if frame_idx % args.frame_skip != 0:
            out.write(frame)  # Write original frame
            frame_idx += 1
            pbar.update(1)
            continue
        
        result_frame = frame.copy()
        img_rgb = frame[:, :, ::-1]  # BGR to RGB
        
        try:
            # Detect humans
            det_out = detector(frame)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.3)  # Lower threshold
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            
            if len(pred_bboxes) > 0:
                # Detect keypoints
                vitposes_out = cpm.predict_pose(
                    img_rgb,
                    [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                )
                
                # Extract hand bboxes
                bboxes = []
                is_right = []
                keypoints_2d_list = []
                
                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                    right_hand_keyp = vitposes['keypoints'][-21:]
                    
                    # Left hand
                    keyp = left_hand_keyp
                    valid = keyp[:, 2] > 0.1
                    if sum(valid) > 3:
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                                keyp[valid, 0].max(), keyp[valid, 1].max()]
                        bboxes.append(bbox)
                        is_right.append(0)
                        keypoints_2d_list.append(left_hand_keyp)
                    
                    # Right hand
                    keyp = right_hand_keyp
                    valid = keyp[:, 2] > 0.1
                    if sum(valid) > 3:
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                                keyp[valid, 0].max(), keyp[valid, 1].max()]
                        bboxes.append(bbox)
                        is_right.append(1)
                        keypoints_2d_list.append(right_hand_keyp)
                
                if len(bboxes) > 0:
                    # Draw using ViTPose 2D keypoints directly (more reliable visualization)
                    for hand_idx in range(len(keypoints_2d_list)):
                        is_right_hand = is_right[hand_idx]
                        kp2d = keypoints_2d_list[hand_idx]
                        result_frame = draw_hand_2d(result_frame, kp2d, is_right=is_right_hand, confidence_threshold=0.1)
        
        except Exception as e:
            print(f"Frame {frame_idx}: Error - {e}")
        
        out.write(result_frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\nSaved output video: {args.output_video}")


def main():
    parser = argparse.ArgumentParser(description='Hamba video hand extraction')
    parser.add_argument('--input_video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_video', type=str, required=True, help='Output video path')
    parser.add_argument('--checkpoint', type=str, default="ckpts/hamba/checkpoints/hamba.ckpt")
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--frame_skip', type=int, default=1, help='Process every nth frame')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames to process')
    args = parser.parse_args()
    
    process_video(args)


if __name__ == '__main__':
    main()
