# Complete A2D2 to YOLOv12 Preprocessing Pipeline
# Based on your actual dataset structure

import os
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

class A2D2YOLOConverter:
    def __init__(self, a2d2_root, output_root):
        self.a2d2_root = Path(a2d2_root)
        self.output_root = Path(output_root)
        
        # Load camera/lidar configuration
        self.cams_lidars_config = self.load_config()
        
        # Load class mappings
        self.semantic_classes = self.load_semantic_classes()
        self.bbox_classes = self.load_bbox_classes()
        
        # Create output directories
        self.setup_output_directories()
    
    def load_config(self):
        """Load camera and lidar configuration"""
        config_path = self.a2d2_root / "cams_lidars.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_semantic_classes(self):
        """Load semantic segmentation class definitions"""
        class_path = self.a2d2_root / "camera_lidar_semantic" / "class_list.json"
        with open(class_path, 'r') as f:
            return json.load(f)
    
    def load_bbox_classes(self):
        """Load 3D bounding box class definitions"""
        class_path = self.a2d2_root / "camera_lidar_semantic_bboxes" / "class_list.json"
        with open(class_path, 'r') as f:
            return json.load(f)
    
    def setup_output_directories(self):
        """Create output directory structure for YOLO format"""
        tasks = ['2d_detection', '3d_detection', 'segmentation']
        splits = ['train', 'val', 'test']
        
        for task in tasks:
            for split in splits:
                (self.output_root / task / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.output_root / task / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def get_camera_matrix(self, camera_name="cam_front_center"):
        """Extract camera intrinsic matrix"""
        cameras = self.cams_lidars_config['cameras']
        for cam in cameras:
            if cam['name'] == camera_name:
                # Extract intrinsic matrix
                fx = cam['CameraMatrix'][0]
                fy = cam['CameraMatrix'][4] 
                cx = cam['CameraMatrix'][2]
                cy = cam['CameraMatrix'][5]
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return None
    
    def rgb_to_class_id(self, rgb_mask):
        """Convert RGB semantic mask to class IDs"""
        class_map = {}
        for class_name, rgb_values in self.semantic_classes.items():
            rgb_tuple = tuple(rgb_values)
            class_map[rgb_tuple] = list(self.semantic_classes.keys()).index(class_name)
        
        # Convert RGB mask to class IDs
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        for rgb_tuple, class_id in class_map.items():
            mask = np.all(rgb_mask == rgb_tuple, axis=2)
            class_mask[mask] = class_id
            
        return class_mask
    
    def project_3d_to_2d(self, point_3d, camera_matrix):
        """Project 3D point to 2D image coordinates"""
        point_3d_homo = np.append(point_3d, 1)
        point_2d_homo = camera_matrix @ point_3d_homo[:3]
        
        if point_2d_homo[2] <= 0:  # Behind camera
            return None
            
        x = point_2d_homo[0] / point_2d_homo[2]
        y = point_2d_homo[1] / point_2d_homo[2]
        return [x, y]
    
    def process_3d_bbox_to_2d(self, bbox_3d, camera_matrix, img_width, img_height):
        """Convert 3D bounding box to 2D YOLO format"""
        # Extract 3D bbox parameters
        center = bbox_3d['center']
        size = bbox_3d['size']  # [length, width, height]
        rotation = bbox_3d.get('rotation', [0, 0, 0])
        
        # Generate 8 corners of 3D bounding box
        l, w, h = size
        corners_3d = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        
        # Apply rotation (simplified - you may need proper rotation matrix)
        # For now, just translate to center
        corners_3d += center
        
        # Project all corners to 2D
        corners_2d = []
        for corner in corners_3d:
            point_2d = self.project_3d_to_2d(corner, camera_matrix)
            if point_2d:
                corners_2d.append(point_2d)
        
        if len(corners_2d) < 4:  # Not enough visible corners
            return None
        
        # Get 2D bounding box from projected corners
        corners_2d = np.array(corners_2d)
        x_min, y_min = corners_2d.min(axis=0)
        x_max, y_max = corners_2d.max(axis=0)
        
        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        # Convert to YOLO format (normalized center, width, height)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
    
    def mask_to_polygons(self, mask, class_id):
        """Convert semantic mask to YOLO polygon format"""
        # Find contours for the specific class
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:  # Valid polygon
                polygon = approx.reshape(-1, 2)
                # Normalize coordinates
                polygon = polygon.astype(np.float32)
                polygons.append(polygon)
        
        return polygons
    
    def process_semantic_segmentation(self):
        """Process semantic segmentation data"""
        print("Processing Semantic Segmentation Data...")
        
        semantic_root = self.a2d2_root / "camera_lidar_semantic"
        output_dir = self.output_root / "segmentation"
        
        all_sequences = list(semantic_root.glob("2018*"))
        
        for i, seq_dir in enumerate(tqdm(all_sequences, desc="Processing sequences")):
            # Determine split (train/val/test)
            if i < len(all_sequences) * 0.7:
                split = "train"
            elif i < len(all_sequences) * 0.85:
                split = "val"
            else:
                split = "test"
            
            camera_dir = seq_dir / "camera" / "cam_front_center"
            label_dir = seq_dir / "label" / "cam_front_center"
            
            for img_file in camera_dir.glob("*.png"):
                # Load image
                img = cv2.imread(str(img_file))
                h, w = img.shape[:2]
                
                # Load corresponding semantic mask
                mask_file = label_dir / img_file.name.replace("camera", "label")
                if not mask_file.exists():
                    continue
                
                mask_rgb = cv2.imread(str(mask_file))
                mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                
                # Convert RGB mask to class IDs
                class_mask = self.rgb_to_class_id(mask_rgb)
                
                # Generate YOLO annotations
                annotations = []
                unique_classes = np.unique(class_mask)
                
                for class_id in unique_classes:
                    if class_id == 0:  # Skip background
                        continue
                    
                    polygons = self.mask_to_polygons(class_mask, class_id)
                    for polygon in polygons:
                        if len(polygon) >= 3:
                            # Normalize coordinates
                            norm_polygon = polygon.copy()
                            norm_polygon[:, 0] /= w
                            norm_polygon[:, 1] /= h
                            
                            # Format as YOLO segmentation
                            coords = norm_polygon.flatten()
                            annotation = f"{class_id} " + " ".join(map(str, coords))
                            annotations.append(annotation)
                
                # Save image and annotations
                output_img_path = output_dir / split / "images" / img_file.name
                output_label_path = output_dir / split / "labels" / img_file.with_suffix('.txt').name
                
                # Copy image
                shutil.copy2(img_file, output_img_path)
                
                # Save annotations
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(annotations))
    
    def process_2d_detection(self):
        """Process 2D object detection data (from 3D boxes)"""
        print("Processing 2D Object Detection Data...")
        
        bbox_root = self.a2d2_root / "camera_lidar_semantic_bboxes"
        output_dir = self.output_root / "2d_detection"
        
        camera_matrix = self.get_camera_matrix()
        all_sequences = list(bbox_root.glob("2018*"))
        
        for i, seq_dir in enumerate(tqdm(all_sequences, desc="Processing sequences")):
            # Determine split
            if i < len(all_sequences) * 0.7:
                split = "train"
            elif i < len(all_sequences) * 0.85:
                split = "val"
            else:
                split = "test"
            
            camera_dir = seq_dir / "camera" / "cam_front_center"
            label3d_dir = seq_dir / "label3D" / "cam_front_center"
            
            for img_file in camera_dir.glob("*.png"):
                # Load image
                img = cv2.imread(str(img_file))
                h, w = img.shape[:2]
                
                # Load corresponding 3D annotations
                label_file = label3d_dir / img_file.name.replace("camera", "label3D").replace(".png", ".json")
                if not label_file.exists():
                    continue
                
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                
                annotations = []
                
                # Process each 3D bounding box
                for bbox_3d in label_data:
                    class_name = bbox_3d.get('class', 'unknown')
                    
                    # Map class name to ID
                    class_names = list(self.bbox_classes.keys())
                    if class_name in class_names:
                        class_id = class_names.index(class_name)
                    else:
                        continue
                    
                    # Convert 3D bbox to 2D YOLO format
                    bbox_2d = self.process_3d_bbox_to_2d(bbox_3d, camera_matrix, w, h)
                    
                    if bbox_2d:
                        annotation = f"{class_id} " + " ".join(map(str, bbox_2d))
                        annotations.append(annotation)
                
                # Save image and annotations
                output_img_path = output_dir / split / "images" / img_file.name
                output_label_path = output_dir / split / "labels" / img_file.with_suffix('.txt').name
                
                # Copy image
                shutil.copy2(img_file, output_img_path)
                
                # Save annotations
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(annotations))
    
    def process_3d_detection(self):
        """Process 3D object detection data"""
        print("Processing 3D Object Detection Data...")
        
        bbox_root = self.a2d2_root / "camera_lidar_semantic_bboxes"
        output_dir = self.output_root / "3d_detection"
        
        all_sequences = list(bbox_root.glob("2018*"))
        
        for i, seq_dir in enumerate(tqdm(all_sequences, desc="Processing sequences")):
            # Determine split
            if i < len(all_sequences) * 0.7:
                split = "train"
            elif i < len(all_sequences) * 0.85:
                split = "val"
            else:
                split = "test"
            
            camera_dir = seq_dir / "camera" / "cam_front_center"
            label3d_dir = seq_dir / "label3D" / "cam_front_center"
            lidar_dir = seq_dir / "lidar" / "cam_front_center"
            
            for img_file in camera_dir.glob("*.png"):
                # Load image
                img = cv2.imread(str(img_file))
                h, w = img.shape[:2]
                
                # Load corresponding 3D annotations
                label_file = label3d_dir / img_file.name.replace("camera", "label3D").replace(".png", ".json")
                lidar_file = lidar_dir / img_file.name.replace("camera", "lidar").replace(".png", ".npz")
                
                if not label_file.exists() or not lidar_file.exists():
                    continue
                
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                
                # Load LiDAR data
                lidar_data = np.load(lidar_file)
                
                annotations = []
                
                # Process each 3D bounding box
                for bbox_3d in label_data:
                    class_name = bbox_3d.get('class', 'unknown')
                    
                    # Map class name to ID
                    class_names = list(self.bbox_classes.keys())
                    if class_name in class_names:
                        class_id = class_names.index(class_name)
                    else:
                        continue
                    
                    # Extract 3D bbox parameters
                    center = bbox_3d['center']
                    size = bbox_3d['size']
                    rotation = bbox_3d.get('rotation', [0, 0, 0])
                    
                    # Format for 3D YOLO (you may need to adapt this format)
                    annotation = f"{class_id} {center[0]} {center[1]} {center[2]} {size[0]} {size[1]} {size[2]} {rotation[2]}"
                    annotations.append(annotation)
                
                # Save image, LiDAR, and annotations
                output_img_path = output_dir / split / "images" / img_file.name
                output_label_path = output_dir / split / "labels" / img_file.with_suffix('.txt').name
                output_lidar_path = output_dir / split / "lidar" / lidar_file.name
                
                # Create lidar directory
                (output_dir / split / "lidar").mkdir(exist_ok=True)
                
                # Copy files
                shutil.copy2(img_file, output_img_path)
                shutil.copy2(lidar_file, output_lidar_path)
                
                # Save annotations
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(annotations))
    
    def create_dataset_configs(self):
        """Create YOLO dataset configuration files"""
        configs = {
            'segmentation': {
                'path': str(self.output_root / 'segmentation'),
                'train': 'train/images',
                'val': 'val/images', 
                'test': 'test/images',
                'nc': len(self.semantic_classes),
                'names': list(self.semantic_classes.keys())
            },
            '2d_detection': {
                'path': str(self.output_root / '2d_detection'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images', 
                'nc': len(self.bbox_classes),
                'names': list(self.bbox_classes.keys())
            },
            '3d_detection': {
                'path': str(self.output_root / '3d_detection'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.bbox_classes),
                'names': list(self.bbox_classes.keys()),
                'depth_range': [0, 80],
                'anchor_sizes': [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
            }
        }
        
        for task, config in configs.items():
            config_path = self.output_root / f'{task}_config.yaml'
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def run_conversion(self):
        """Run the complete conversion process"""
        print("Starting A2D2 to YOLO conversion...")
        print(f"Input: {self.a2d2_root}")
        print(f"Output: {self.output_root}")
        
        # Process all tasks
        self.process_semantic_segmentation()
        self.process_2d_detection()
        self.process_3d_detection()
        
        # Create dataset configs
        self.create_dataset_configs()
        
        print("Conversion completed!")
        print(f"Dataset configs saved to: {self.output_root}")

if __name__ == "__main__":
    # Configuration
    A2D2_ROOT = "C:/"  # Your A2D2 dataset root directory
    OUTPUT_ROOT = "C:/a2d2_yolo"  # Output directory for YOLO format
    
    # Run conversion
    converter = A2D2YOLOConverter(A2D2_ROOT, OUTPUT_ROOT)
    converter.run_conversion()
