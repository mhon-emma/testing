#!/usr/bin/env python3
"""
Proper A2D2 to YOLOv12 Converter
Supports: 2D Detection, 3D Detection (with modifications), Semantic Segmentation
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml

class ProperA2D2YOLOConverter:
    def __init__(self):
        self.a2d2_root = Path("/home/Lambdaone/Emma/a2d2_full")
        self.output_root = Path("/home/Lambdaone/Emma/a2d2_yolo")
        
        # Load configurations
        self.cams_lidars_config = self.load_config()
        self.semantic_classes = self.load_semantic_classes()
        self.bbox_classes = self.load_bbox_classes()
        
        # Camera calibration
        self.camera_matrix = self.get_camera_matrix()
        
        self.setup_output_directories()
    
    def load_config(self):
        """Load camera and lidar configuration"""
        config_path = self.a2d2_root / "cams_lidars.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def load_semantic_classes(self):
        """Load semantic class definitions with proper RGB mapping"""
        class_path = self.a2d2_root / "camera_lidar_semantic" / "class_list.json"
        try:
            with open(class_path, 'r') as f:
                return json.load(f)
        except:
            # A2D2 standard classes (you may need to verify these)
            return {
                "Car": [255, 0, 0],
                "Pedestrian": [0, 255, 0],
                "Bicycle": [0, 0, 255],
                "Road": [128, 128, 128],
                "Sidewalk": [244, 35, 232],
                "Building": [70, 70, 70],
                "Vegetation": [107, 142, 35],
                "Sky": [70, 130, 180],
                "Pole": [153, 153, 153],
                "TrafficSign": [220, 220, 0],
                "TrafficLight": [250, 170, 30]
            }
    
    def load_bbox_classes(self):
        """Load 3D bounding box class definitions"""
        class_path = self.a2d2_root / "camera_lidar_semantic_bboxes" / "class_list.json"
        try:
            with open(class_path, 'r') as f:
                return json.load(f)
        except:
            return ["Car", "Pedestrian", "Bicycle", "Bus", "Truck", "TrafficSign", "TrafficLight"]
    
    def get_camera_matrix(self):
        """Extract proper camera intrinsic matrix"""
        if not self.cams_lidars_config:
            return None
            
        cameras = self.cams_lidars_config.get('cameras', [])
        for cam in cameras:
            if cam.get('name') == 'cam_front_center':
                matrix = cam.get('CameraMatrix', [])
                if len(matrix) >= 9:
                    K = np.array([
                        [matrix[0], matrix[1], matrix[2]],
                        [matrix[3], matrix[4], matrix[5]],
                        [matrix[6], matrix[7], matrix[8]]
                    ])
                    return K
        return None
    
    def setup_output_directories(self):
        """Create output directories"""
        tasks = ['2d_detection', '3d_detection', 'segmentation']
        splits = ['train', 'val', 'test']
        
        for task in tasks:
            for split in splits:
                (self.output_root / task / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.output_root / task / split / 'labels').mkdir(parents=True, exist_ok=True)
                
                if task == '3d_detection':
                    (self.output_root / task / split / 'point_clouds').mkdir(parents=True, exist_ok=True)
    
    def project_3d_to_2d(self, point_3d, camera_matrix):
        """Properly project 3D point to 2D image coordinates"""
        if camera_matrix is None:
            return None
            
        # Convert to homogeneous coordinates
        point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        
        # Project to camera coordinates (you may need extrinsic matrix here)
        # For now, assuming points are already in camera coordinate system
        point_cam = point_3d
        
        if point_cam[2] <= 0:  # Behind camera
            return None
        
        # Project to image plane
        point_2d_homo = camera_matrix @ point_cam
        
        if point_2d_homo[2] == 0:
            return None
            
        x = point_2d_homo[0] / point_2d_homo[2]
        y = point_2d_homo[1] / point_2d_homo[2]
        
        return [x, y]
    
    def convert_3d_bbox_to_2d(self, bbox_3d, camera_matrix, img_width, img_height):
        """Convert 3D bounding box to 2D YOLO format"""
        if camera_matrix is None:
            return None
            
        # Extract 3D box parameters
        center = bbox_3d.get('center', [0, 0, 0])
        size = bbox_3d.get('size', [1, 1, 1])
        rotation = bbox_3d.get('rotation', [0, 0, 0])
        
        # Generate 8 corners of 3D bounding box
        l, w, h = size[0], size[1], size[2]
        
        # Box corners in object coordinate system
        corners = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        
        # Apply rotation (simplified - implement proper rotation matrix if needed)
        # For now, just translate to center
        corners += center
        
        # Project all corners to 2D
        projected_corners = []
        for corner in corners:
            point_2d = self.project_3d_to_2d(corner, camera_matrix)
            if point_2d is not None:
                projected_corners.append(point_2d)
        
        if len(projected_corners) < 4:
            return None
        
        # Get 2D bounding box from projected corners
        projected_corners = np.array(projected_corners)
        x_min = max(0, projected_corners[:, 0].min())
        y_min = max(0, projected_corners[:, 1].min())
        x_max = min(img_width, projected_corners[:, 0].max())
        y_max = min(img_height, projected_corners[:, 1].max())
        
        # Convert to YOLO format (normalized)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Filter out invalid boxes
        if width <= 0 or height <= 0 or width > 1 or height > 1:
            return None
            
        return [x_center, y_center, width, height]
    
    def rgb_to_class_mask(self, rgb_image):
        """Convert RGB semantic image to class mask"""
        if rgb_image is None or len(rgb_image.shape) != 3:
            return np.zeros((100, 100), dtype=np.uint8)
        
        h, w = rgb_image.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        for class_id, (class_name, rgb_value) in enumerate(self.semantic_classes.items()):
            if isinstance(rgb_value, list) and len(rgb_value) == 3:
                r, g, b = rgb_value
                mask = (rgb_image[:, :, 0] == r) & (rgb_image[:, :, 1] == g) & (rgb_image[:, :, 2] == b)
                class_mask[mask] = class_id
        
        return class_mask
    
    def mask_to_yolo_polygons(self, class_mask, class_id, img_width, img_height):
        """Convert class mask to YOLO polygon format"""
        binary_mask = (class_mask == class_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                # Normalize coordinates
                polygon = approx.reshape(-1, 2).astype(np.float32)
                polygon[:, 0] /= img_width
                polygon[:, 1] /= img_height
                
                # Clip to [0, 1]
                polygon = np.clip(polygon, 0, 1)
                
                # Format for YOLO
                coords = polygon.flatten()
                polygons.append(coords)
        
        return polygons
    
    def process_2d_detection(self):
        """Process 2D object detection with proper 3D to 2D projection"""
        print("Processing 2D Object Detection...")
        
        bbox_root = self.a2d2_root / "camera_lidar_semantic_bboxes"
        sequences = list(bbox_root.glob("2018*"))
        
        if not sequences:
            print("[ERROR] No bbox sequences found")
            return
        
        # Split sequences
        n_train = int(len(sequences) * 0.7)
        n_val = int(len(sequences) * 0.15)
        
        splits = [
            (sequences[:n_train], "train"),
            (sequences[n_train:n_train+n_val], "val"),
            (sequences[n_train+n_val:], "test")
        ]
        
        total_processed = 0
        
        for seq_list, split_name in splits:
            if not seq_list:
                continue
                
            print(f"Processing {split_name} split...")
            
            for seq_dir in tqdm(seq_list, desc=f"{split_name}"):
                camera_dir = seq_dir / "camera" / "cam_front_center"
                label3d_dir = seq_dir / "label3D" / "cam_front_center"
                
                for img_file in camera_dir.glob("*.png"):
                    label_file = label3d_dir / img_file.name.replace("camera", "label3D").replace(".png", ".json")
                    
                    if not label_file.exists():
                        continue
                    
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Load 3D annotations
                    try:
                        with open(label_file, 'r') as f:
                            annotations_3d = json.load(f)
                    except:
                        continue
                    
                    yolo_annotations = []
                    
                    # Process each 3D bounding box
                    for bbox_3d in annotations_3d:
                        class_name = bbox_3d.get('class', 'unknown')
                        
                        # Map class name to ID
                        if class_name in self.bbox_classes:
                            class_id = self.bbox_classes.index(class_name)
                        else:
                            continue
                        
                        # Convert 3D box to 2D
                        bbox_2d = self.convert_3d_bbox_to_2d(bbox_3d, self.camera_matrix, w, h)
                        
                        if bbox_2d is not None:
                            annotation = f"{class_id} {bbox_2d[0]:.6f} {bbox_2d[1]:.6f} {bbox_2d[2]:.6f} {bbox_2d[3]:.6f}"
                            yolo_annotations.append(annotation)
                    
                    # Save files
                    out_img = self.output_root / "2d_detection" / split_name / "images" / img_file.name
                    out_label = self.output_root / "2d_detection" / split_name / "labels" / img_file.with_suffix('.txt').name
                    
                    shutil.copy2(img_file, out_img)
                    
                    with open(out_label, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    total_processed += 1
        
        print(f"[OK] 2D detection: {total_processed} images processed")
    
    def process_semantic_segmentation(self):
        """Process semantic segmentation with proper polygon conversion"""
        print("Processing Semantic Segmentation...")
        
        semantic_root = self.a2d2_root / "camera_lidar_semantic"
        sequences = list(semantic_root.glob("2018*"))
        
        if not sequences:
            print("[ERROR] No semantic sequences found")
            return
        
        # Split sequences
        n_train = int(len(sequences) * 0.7)
        n_val = int(len(sequences) * 0.15)
        
        splits = [
            (sequences[:n_train], "train"),
            (sequences[n_train:n_train+n_val], "val"),
            (sequences[n_train+n_val:], "test")
        ]
        
        total_processed = 0
        
        for seq_list, split_name in splits:
            if not seq_list:
                continue
                
            print(f"Processing {split_name} split...")
            
            for seq_dir in tqdm(seq_list, desc=f"{split_name}"):
                camera_dir = seq_dir / "camera" / "cam_front_center"
                label_dir = seq_dir / "label" / "cam_front_center"
                
                for img_file in camera_dir.glob("*.png"):
                    label_file = label_dir / img_file.name.replace("camera", "label")
                    
                    if not label_file.exists():
                        continue
                    
                    # Load image and mask
                    img = cv2.imread(str(img_file))
                    mask_img = cv2.imread(str(label_file))
                    
                    if img is None or mask_img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Convert BGR to RGB
                    mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                    
                    # Convert to class mask
                    class_mask = self.rgb_to_class_mask(mask_rgb)
                    
                    # Generate YOLO segmentation annotations
                    yolo_annotations = []
                    unique_classes = np.unique(class_mask)
                    
                    for class_id in unique_classes:
                        if class_id == 0:  # Skip background
                            continue
                        
                        # Convert mask to polygons
                        polygons = self.mask_to_yolo_polygons(class_mask, class_id, w, h)
                        
                        for polygon in polygons:
                            if len(polygon) >= 6:  # At least 3 points
                                coords_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                                annotation = f"{class_id} {coords_str}"
                                yolo_annotations.append(annotation)
                    
                    # Save files
                    out_img = self.output_root / "segmentation" / split_name / "images" / img_file.name
                    out_label = self.output_root / "segmentation" / split_name / "labels" / img_file.with_suffix('.txt').name
                    
                    shutil.copy2(img_file, out_img)
                    
                    with open(out_label, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    total_processed += 1
        
        print(f"[OK] Segmentation: {total_processed} images processed")
    
    def process_3d_detection(self):
        """Process 3D detection data (for specialized 3D models)"""
        print("Processing 3D Detection...")
        print("[WARNING] Standard YOLOv12 doesn't support 3D detection natively")
        print("[INFO] Preparing data for 3D-aware models or future extensions")
        
        bbox_root = self.a2d2_root / "camera_lidar_semantic_bboxes"
        sequences = list(bbox_root.glob("2018*"))
        
        if not sequences:
            print("[ERROR] No bbox sequences found")
            return
        
        # Split sequences
        n_train = int(len(sequences) * 0.7)
        n_val = int(len(sequences) * 0.15)
        
        splits = [
            (sequences[:n_train], "train"),
            (sequences[n_train:n_train+n_val], "val"),
            (sequences[n_train+n_val:], "test")
        ]
        
        total_processed = 0
        
        for seq_list, split_name in splits:
            if not seq_list:
                continue
                
            print(f"Processing {split_name} split...")
            
            for seq_dir in tqdm(seq_list, desc=f"{split_name}"):
                camera_dir = seq_dir / "camera" / "cam_front_center"
                label3d_dir = seq_dir / "label3D" / "cam_front_center"
                lidar_dir = seq_dir / "lidar" / "cam_front_center"
                
                for img_file in camera_dir.glob("*.png"):
                    label_file = label3d_dir / img_file.name.replace("camera", "label3D").replace(".png", ".json")
                    lidar_file = lidar_dir / img_file.name.replace("camera", "lidar").replace(".png", ".npz")
                    
                    if not label_file.exists() or not lidar_file.exists():
                        continue
                    
                    try:
                        # Load 3D annotations
                        with open(label_file, 'r') as f:
                            annotations_3d = json.load(f)
                        
                        # Load LiDAR data
                        lidar_data = np.load(lidar_file)
                        
                        # Process 3D annotations
                        yolo_3d_annotations = []
                        
                        for bbox_3d in annotations_3d:
                            class_name = bbox_3d.get('class', 'unknown')
                            
                            if class_name in self.bbox_classes:
                                class_id = self.bbox_classes.index(class_name)
                            else:
                                continue
                            
                            center = bbox_3d.get('center', [0, 0, 0])
                            size = bbox_3d.get('size', [1, 1, 1])
                            rotation = bbox_3d.get('rotation', [0, 0, 0])
                            
                            # Format: class_id x y z l w h rotation_y
                            annotation = f"{class_id} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} {size[0]:.6f} {size[1]:.6f} {size[2]:.6f} {rotation[2]:.6f}"
                            yolo_3d_annotations.append(annotation)
                        
                        # Save files
                        out_img = self.output_root / "3d_detection" / split_name / "images" / img_file.name
                        out_label = self.output_root / "3d_detection" / split_name / "labels" / img_file.with_suffix('.txt').name
                        out_lidar = self.output_root / "3d_detection" / split_name / "point_clouds" / lidar_file.name
                        
                        shutil.copy2(img_file, out_img)
                        shutil.copy2(lidar_file, out_lidar)
                        
                        with open(out_label, 'w') as f:
                            f.write('\n'.join(yolo_3d_annotations))
                        
                        total_processed += 1
                        
                    except Exception as e:
                        continue
        
        print(f"[OK] 3D detection: {total_processed} images processed")
    
    def create_configs(self):
        """Create dataset configuration files"""
        configs = {
            '2d_detection_config.yaml': {
                'path': str(self.output_root / '2d_detection'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.bbox_classes),
                'names': self.bbox_classes
            },
            'segmentation_config.yaml': {
                'path': str(self.output_root / 'segmentation'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.semantic_classes),
                'names': list(self.semantic_classes.keys())
            },
            '3d_detection_config.yaml': {
                'path': str(self.output_root / '3d_detection'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.bbox_classes),
                'names': self.bbox_classes,
                'point_clouds': True,
                'format': '3d'
            }
        }
        
        for config_name, config_data in configs.items():
            config_path = self.output_root / config_name
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"[OK] Created: {config_path}")
    
    def run_conversion(self):
        """Run the complete conversion process"""
        print("Starting Proper A2D2 to YOLO conversion...")
        
        # Process all tasks
        self.process_2d_detection()
        self.process_semantic_segmentation()
        self.process_3d_detection()
        
        # Create configs
        self.create_configs()
        
        print("\nConversion completed!")
        print(f"Data saved to: {self.output_root}")

def main():
    converter = ProperA2D2YOLOConverter()
    converter.run_conversion()

if __name__ == "__main__":
    main()