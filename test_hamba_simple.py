"""
Simple test script for Hamba - no rendering, just inference and save results
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

from hamba.configs import CACHE_DIR_HAMBA, get_config
from hamba.models import MANO
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


def load_hamba_no_renderer(checkpoint_path):
    """Load Hamba model without initializing the renderer"""
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)
    
    # Override some config values
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    
    # Remove pretrained weights config that causes issues
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


def draw_hand_skeleton(img, keypoints_3d, cam_t, focal_length, img_center, is_right=True, color=(0, 255, 0)):
    """Draw hand skeleton on image using 3D to 2D projection"""
    h, w = img.shape[:2]
    
    # Project 3D points to 2D
    points_2d = perspective_project(keypoints_3d, cam_t, focal_length, img_center)
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points_2d) and end_idx < len(points_2d):
            pt1 = tuple(points_2d[start_idx].astype(int))
            pt2 = tuple(points_2d[end_idx].astype(int))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, color, 2)
    
    # Draw joints
    for pt in points_2d:
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
            cv2.circle(img, pt_int, 4, (0, 0, 255), -1)
    
    return img


def perspective_project(points_3d, cam_t, focal_length, img_center):
    """Project 3D points to 2D using perspective projection"""
    # Translate points
    points_cam = points_3d + cam_t
    
    # Perspective projection
    x = points_cam[:, 0] / points_cam[:, 2] * focal_length + img_center[0]
    y = points_cam[:, 1] / points_cam[:, 2] * focal_length + img_center[1]
    
    return np.stack([x, y], axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Hamba simple test (no rendering)')
    parser.add_argument('--img_folder', type=str, default='./example_data', help='Folder with input images')
    parser.add_argument('--checkpoint', type=str, default="ckpts/hamba/checkpoints/hamba.ckpt")
    parser.add_argument('--out_folder', type=str, default='./demo_out/', help='Output folder')
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    args = parser.parse_args()
    
    # Load model without renderer
    model, model_cfg = load_hamba_no_renderer(args.checkpoint)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Load detector
    from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamba
    cfg_path = Path(hamba.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.1
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # Keypoint detector
    cpm = ViTPoseModel(device)
    
    # Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    
    # Get all images
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)
    
    print(f"Found {len(img_paths)} images")
    
    for img_path in img_paths:
        print(f"\nProcessing: {img_path}")
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"  Could not read image")
            continue
            
        img = img_cv2.copy()[:, :, ::-1]  # BGR to RGB
        
        # Detect humans
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        print(f"  Detected {len(pred_bboxes)} people")
        
        if len(pred_bboxes) == 0:
            cv2.imwrite(os.path.join(args.out_folder, f'{img_path.stem}_no_detection.jpg'), img_cv2)
            continue
        
        # Detect keypoints
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        # Extract hand bboxes from keypoints
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
        
        print(f"  Detected {len(bboxes)} hands")
        
        if len(bboxes) == 0:
            cv2.imwrite(os.path.join(args.out_folder, f'{img_path.stem}_no_hands.jpg'), img_cv2)
            continue
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        keypoints_2d_arr = np.stack(keypoints_2d_list)
        
        # Run Hamba inference
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, 
                               rescale_factor=args.rescale_factor, 
                               keypoints_2d_arr=keypoints_2d_arr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        result_img = img_cv2.copy()
        all_results = []
        
        for batch in dataloader:
            if -1 in batch["is_valid"]:
                print("  Invalid batch")
                continue
            
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            
            # Get predictions
            pred_vertices = out['pred_vertices'].cpu().numpy()
            pred_keypoints_3d = out['pred_keypoints_3d'].cpu().numpy()
            pred_cam = out['pred_cam'].cpu().numpy()
            
            # Get camera translation
            multiplier = (2 * batch['right'] - 1).cpu().numpy()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float().cpu().numpy()
            box_size = batch["box_size"].float().cpu().numpy()
            img_size = batch["img_size"].float().cpu().numpy()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            
            pred_cam_t = cam_crop_to_full(
                torch.from_numpy(pred_cam), 
                torch.from_numpy(box_center), 
                torch.from_numpy(box_size),
                torch.from_numpy(img_size), 
                scaled_focal_length
            ).numpy()
            
            batch_size = pred_vertices.shape[0]
            for n in range(batch_size):
                is_right_hand = batch['right'][n].cpu().item()
                verts = pred_vertices[n]
                joints = pred_keypoints_3d[n]
                cam_t = pred_cam_t[n]
                
                # Flip x for right hand
                if is_right_hand:
                    verts[:, 0] = -verts[:, 0]
                    joints[:, 0] = -joints[:, 0]
                
                # Draw skeleton on image
                color = (0, 255, 0) if is_right_hand else (255, 0, 0)  # Green for right, Blue for left
                img_center = np.array([img_cv2.shape[1] / 2, img_cv2.shape[0] / 2])
                result_img = draw_hand_skeleton(result_img, joints, cam_t, 
                                                scaled_focal_length, img_center,
                                                is_right=is_right_hand, color=color)
                
                # Store results
                all_results.append({
                    'is_right': bool(is_right_hand),
                    'vertices': verts.tolist(),
                    'joints_3d': joints.tolist(),
                    'cam_t': cam_t.tolist(),
                })
        
        # Save result image
        out_path = os.path.join(args.out_folder, f'{img_path.stem}_hands.jpg')
        cv2.imwrite(out_path, result_img)
        print(f"  Saved: {out_path}")
        
        # Save JSON results
        import json
        json_path = os.path.join(args.out_folder, f'{img_path.stem}_hands.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved: {json_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
