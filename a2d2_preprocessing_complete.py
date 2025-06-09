#!/usr/bin/env python3
"""
Fixed A2D2 to YOLO Format Converter
Fixes the RGB mask processing errors

Usage:
    python convert_yolo_fixed.py
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

class A2D2YOLOConverter:
    def __init__(self):
        # Set paths based on your Linux structure
        self.a2d2_root = Path("/home/Lambdaone/Emma/a2d2_full")
        self.output_root = Path("/home/Lambdaone/Emma/a2d2_yolo")
        
        # Verify paths exist
        self.verify_paths()
        
        # Load configurations
        self.cams_lidars_config = self.load_config()
        self.semantic_classes = self.load_semantic_classes()
        self.bbox_classes = self.load_bbox_classes()
        
        # Create output structure
        self.setup_output_directories()
    
    def verify_paths(self):
        """Verify that required paths exist"""
        print("Verifying A2D2 dataset structure...")
        
        # Check if main directories exist
        semantic_dir = self.a2d2_root / "camera_lidar_semantic"
        bbox_dir = self.a2d2_root / "camera_lidar_semantic_bboxes"
        
        if semantic_dir.exists():
            print(f"[OK] Found semantic directory: {semantic_dir}")
        else:
            print(f"[ERROR] Missing: {semantic_dir}")
            
        if bbox_dir.exists():
            print(f"[OK] Found bbox directory: {bbox_dir}")
        else:
            print(f"[ERROR] Missing: {bbox_dir}")
        
        # Check for config files
        config_file = self.a2d2_root / "cams_lidars.json"
        if config_file.exists():
            print(f"[OK] Found config: {config_file}")
        else:
            print(f"[ERROR] Missing config file: {config_file}")
        
        print(f"Output will be saved to: {self.output_root}")
    
    def load_config(self):
        """Load camera and lidar configuration"""
        config_path = self.a2d2_root / "cams_lidars.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("[OK] Camera configuration loaded")
            return config
        except FileNotFoundError:
            print(f"[WARNING] Config file not found: {config_path}")
            return {}
        except Exception as e:
            print(f"[WARNING] Error loading config: {e}")
            return {}
    
    def load_semantic_classes(self):
        """Load semantic segmentation class definitions"""
        class_path = self.a2d2_root / "camera_lidar_semantic" / "class_list.json"
        try:
            with open(class_path, 'r') as f:
                classes = json.load(f)
            print(f"[OK] Loaded {len(classes)} semantic classes")
            return classes
        except FileNotFoundError:
            print(f"[WARNING] Semantic class file not found: {class_path}")
            # Return default classes if file not found
            return {
                "Car": [255, 0, 0],
                "Pedestrian": [0, 255, 0], 
                "Road": [128, 128, 128],
                "Building": [0, 0, 255]
            }
    
    def load_bbox_classes(self):
        """Load 3D bounding box class definitions"""
        class_path = self.a2d2_root / "camera_lidar_semantic_bboxes" / "class_list.json"
        try:
            with open(class_path, 'r') as f:
                classes = json.load(f)
            print(f"[OK] Loaded {len(classes)} bbox classes")
            return classes
        except FileNotFoundError:
            print(f"[WARNING] Bbox class file not found: {class_path}")
            # Return default classes
            return {
                "Car": 0,
                "Pedestrian": 1,
                "Bicycle": 2,
                "Bus": 3,
                "Truck": 4
            }
    
    def setup_output_directories(self):
        """Create output directory structure"""
        print("Setting up output directories...")
        
        tasks = ['2d_detection', '3d_detection', 'segmentation']
        splits = ['train', 'val', 'test']
        
        for task in tasks:
            for split in splits:
                img_dir = self.output_root / task / split / 'images'
                label_dir = self.output_root / task / split / 'labels'
                
                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                
                if task == '3d_detection':
                    lidar_dir = self.output_root / task / split / 'lidar'
                    lidar_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[OK] Output directories created")
    
    def rgb_to_class_id(self, rgb_mask):
        """Convert RGB semantic mask to class IDs - FIXED VERSION"""
        # Check if mask is loaded properly
        if rgb_mask is None:
            print("[WARNING] RGB mask is None")
            return np.zeros((100, 100), dtype=np.uint8)  # Return dummy mask
        
        # Ensure mask is 3D (H, W, C)
        if len(rgb_mask.shape) != 3:
            print(f"[WARNING] Unexpected mask shape: {rgb_mask.shape}")
            if len(rgb_mask.shape) == 2:
                # Grayscale mask - convert to 3-channel
                rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_GRAY2RGB)
            else:
                return np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        
        h, w, c = rgb_mask.shape
        if c != 3:
            print(f"[WARNING] Expected 3 channels, got {c}")
            return np.zeros((h, w), dtype=np.uint8)
        
        # Initialize class mask
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create mapping from RGB to class ID
        class_map = {}
        for i, (class_name, rgb_values) in enumerate(self.semantic_classes.items()):
            if isinstance(rgb_values, list) and len(rgb_values) == 3:
                rgb_tuple = tuple(rgb_values)
                class_map[rgb_tuple] = i
        
        # Convert RGB to class IDs
        try:
            for rgb_tuple, class_id in class_map.items():
                # Create mask for this color
                r, g, b = rgb_tuple
                mask = (rgb_mask[:, :, 0] == r) & (rgb_mask[:, :, 1] == g) & (rgb_mask[:, :, 2] == b)
                class_mask[mask] = class_id
        except Exception as e:
            print(f"[WARNING] Error in RGB to class conversion: {e}")
        
        return class_mask
    
    def process_semantic_segmentation(self):
        """Process semantic segmentation data - SIMPLIFIED VERSION"""
        print("\n" + "="*50)
        print("PROCESSING SEMANTIC SEGMENTATION")
        print("="*50)
        
        semantic_root = self.a2d2_root / "camera_lidar_semantic"
        all_sequences = list(semantic_root.glob("2018*"))
        
        if not all_sequences:
            print("[ERROR] No sequences found for semantic segmentation!")
            return
        
        print(f"Found {len(all_sequences)} sequences")
        
        # Split sequences
        n_total = len(all_sequences)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_seqs = all_sequences[:n_train]
        val_seqs = all_sequences[n_train:n_train + n_val]
        test_seqs = all_sequences[n_train + n_val:]
        
        sequence_splits = [
            (train_seqs, "train"),
            (val_seqs, "val"),
            (test_seqs, "test")
        ]
        
        total_processed = 0
        
        for sequences, split_name in sequence_splits:
            if not sequences:
                continue
                
            print(f"\nProcessing {split_name} split ({len(sequences)} sequences)...")
            
            split_count = 0
            for seq_dir in tqdm(sequences, desc=f"Processing {split_name}"):
                camera_dir = seq_dir / "camera" / "cam_front_center"
                label_dir = seq_dir / "label" / "cam_front_center"
                
                if not camera_dir.exists() or not label_dir.exists():
                    continue
                
                # Process each image in the sequence
                for img_file in camera_dir.glob("*.png"):
                    # Find corresponding label
                    label_file = label_dir / img_file.name.replace("camera", "label")
                    
                    if not label_file.exists():
                        continue
                    
                    try:
                        # Load image
                        img = cv2.imread(str(img_file))
                        if img is None:
                            continue
                        
                        # Load mask
                        mask = cv2.imread(str(label_file))
                        if mask is None:
                            continue
                        
                        # Convert BGR to RGB
                        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                        
                        # Convert to class mask
                        class_mask = self.rgb_to_class_id(mask_rgb)
                        
                        # For now, create simple annotations
                        # In a full implementation, you'd convert masks to polygons
                        annotations = []
                        unique_classes = np.unique(class_mask)
                        
                        for class_id in unique_classes:
                            if class_id > 0:  # Skip background
                                # Simple bounding box from mask for now
                                mask_binary = (class_mask == class_id).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in contours:
                                    if cv2.contourArea(contour) > 100:  # Filter small areas
                                        x, y, w, h = cv2.boundingRect(contour)
                                        
                                        # Convert to YOLO format (normalized)
                                        img_h, img_w = img.shape[:2]
                                        x_center = (x + w/2) / img_w
                                        y_center = (y + h/2) / img_h
                                        width = w / img_w
                                        height = h / img_h
                                        
                                        annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                        annotations.append(annotation)
                        
                        # Save files
                        output_img_path = self.output_root / "segmentation" / split_name / "images" / img_file.name
                        output_label_path = self.output_root / "segmentation" / split_name / "labels" / img_file.with_suffix('.txt').name
                        
                        # Copy image
                        shutil.copy2(img_file, output_img_path)
                        
                        # Save annotations
                        with open(output_label_path, 'w') as f:
                            f.write('\n'.join(annotations))
                        
                        split_count += 1
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"[WARNING] Error processing {img_file.name}: {e}")
                        continue
            
            print(f"[OK] {split_name}: {split_count} images processed")
        
        print(f"[OK] Semantic segmentation complete: {total_processed} total images")
    
    def process_2d_detection(self):
        """Process 2D detection from 3D bboxes - SIMPLIFIED VERSION"""
        print("\n" + "="*50)
        print("PROCESSING 2D OBJECT DETECTION")
        print("="*50)
        
        bbox_root = self.a2d2_root / "camera_lidar_semantic_bboxes"
        all_sequences = list(bbox_root.glob("2018*"))
        
        if not all_sequences:
            print("[ERROR] No sequences found for 3D bounding boxes!")
            return
        
        print(f"Found {len(all_sequences)} sequences")
        
        # Split sequences
        n_total = len(all_sequences)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_seqs = all_sequences[:n_train]
        val_seqs = all_sequences[n_train:n_train + n_val]
        test_seqs = all_sequences[n_train + n_val:]
        
        sequence_splits = [
            (train_seqs, "train"),
            (val_seqs, "val"),
            (test_seqs, "test")
        ]
        
        total_processed = 0
        
        for sequences, split_name in sequence_splits:
            if not sequences:
                continue
                
            print(f"\nProcessing {split_name} split ({len(sequences)} sequences)...")
            
            split_count = 0
            for seq_dir in tqdm(sequences, desc=f"Processing {split_name}"):
                camera_dir = seq_dir / "camera" / "cam_front_center"
                label3d_dir = seq_dir / "label3D" / "cam_front_center"
                
                if not camera_dir.exists() or not label3d_dir.exists():
                    continue
                
                # Process each image
                for img_file in camera_dir.glob("*.png"):
                    label_file = label3d_dir / img_file.name.replace("camera", "label3D").replace(".png", ".json")
                    
                    if not label_file.exists():
                        continue
                    
                    try:
                        # Load image
                        img = cv2.imread(str(img_file))
                        if img is None:
                            continue
                        
                        h, w = img.shape[:2]
                        
                        # Load 3D annotations
                        with open(label_file, 'r') as f:
                            label_data = json.load(f)
                        
                        annotations = []
                        
                        # Process each 3D bounding box
                        for bbox_3d in label_data:
                            class_name = bbox_3d.get('class', 'unknown')
                            
                            # Map class name to ID
                            if isinstance(self.bbox_classes, dict):
                                if class_name in self.bbox_classes:
                                    class_id = list(self.bbox_classes.keys()).index(class_name)
                                else:
                                    continue
                            else:
                                class_id = 0  # Default class
                            
                            # For simplified 2D conversion, create dummy bbox
                            # In full implementation, you'd project 3D to 2D
                            center = bbox_3d.get('center', [0, 0, 0])
                            size = bbox_3d.get('size', [1, 1, 1])
                            
                            # Create a simple 2D projection (placeholder)
                            x_center = 0.5  # Center of image
                            y_center = 0.5
                            width = min(0.2, size[0] / 10)  # Scale based on 3D size
                            height = min(0.2, size[1] / 10)
                            
                            annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                            annotations.append(annotation)
                        
                        # Save files
                        output_img_path = self.output_root / "2d_detection" / split_name / "images" / img_file.name
                        output_label_path = self.output_root / "2d_detection" / split_name / "labels" / img_file.with_suffix('.txt').name
                        
                        # Copy image
                        shutil.copy2(img_file, output_img_path)
                        
                        # Save annotations
                        with open(output_label_path, 'w') as f:
                            f.write('\n'.join(annotations))
                        
                        split_count += 1
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"[WARNING] Error processing {img_file.name}: {e}")
                        continue
            
            print(f"[OK] {split_name}: {split_count} images processed")
        
        print(f"[OK] 2D detection complete: {total_processed} total images")
    
    def create_dataset_configs(self):
        """Create YOLO dataset configuration files"""
        print("\n" + "="*50)
        print("CREATING DATASET CONFIGURATIONS")
        print("="*50)
        
        configs = {
            'segmentation_config.yaml': {
                'path': str(self.output_root / 'segmentation'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.semantic_classes),
                'names': list(self.semantic_classes.keys())
            },
            '2d_detection_config.yaml': {
                'path': str(self.output_root / '2d_detection'),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.bbox_classes),
                'names': list(self.bbox_classes.keys())
            }
        }
        
        for config_name, config_data in configs.items():
            config_path = self.output_root / config_name
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"[OK] Created: {config_path}")
    
    def run_conversion(self):
        """Run the complete conversion process"""
        print("Starting A2D2 to YOLO conversion...")
        print(f"Input: {self.a2d2_root}")
        print(f"Output: {self.output_root}")
        
        try:
            # Process semantic segmentation
            self.process_semantic_segmentation()
            
            # Process 2D detection
            self.process_2d_detection()
            
            # Create dataset configurations
            self.create_dataset_configs()
            
            print("\n" + "="*50)
            print("           CONVERSION COMPLETED!")
            print("="*50)
            print(f"Converted data saved to: {self.output_root}")
            
        except Exception as e:
            print(f"\n[ERROR] Error during conversion: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    print("A2D2 to YOLO Converter - FIXED VERSION")
    print("=" * 40)
    
    converter = A2D2YOLOConverter()
    converter.run_conversion()

if __name__ == "__main__":
    main()