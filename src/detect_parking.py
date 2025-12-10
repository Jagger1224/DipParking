"""
Car Detection and Occupancy Detection
Uses YOLOv8 to detect cars and determines parking spot occupancy

Usage: python src/detect_cars.py
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
FRAMES_DIR = Path("data/frames")
SPOTS_JSON = Path("data/parking_spot_coordinates.json")
OUTPUT_DIR = Path("results/detections")
MODEL_PATH = "yolov8n.pt"  # Will auto-download if not present

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold
IOU_THRESHOLD = 0.3         # Overlap threshold for occupancy

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def load_parking_spots(json_path):
    """Load parking spot polygons from JSON"""
    with open(json_path, 'r') as f:
        spots_data = json.load(f)
    
    spots = []
    for spot_key, spot_info in sorted(spots_data.items()):
        spots.append({
            'id': spot_info['id'],
            'coordinates': spot_info['coordinates']  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        })
    
    return spots


def detect_vehicles(model, image, conf_threshold):
    """
    Run YOLO on image and return vehicle detections
    
    Returns: List of bounding boxes [x1, y1, x2, y2] for detected vehicles
    """
    results = model.predict(image, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        confidence = float(box.conf)
        
        # Only keep vehicle classes (car, motorcycle, bus, truck)
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class': cls_id
            })
    
    return detections


def polygon_to_bbox(polygon):
    """
    Convert polygon to bounding box
    Input: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    Output: [min_x, min_y, max_x, max_y]
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union between two bounding boxes
    Each bbox is [x1, y1, x2, y2]
    """
    # Get intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Check if there's an intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = bbox1_area + bbox2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def determine_occupancy(spots, detections, iou_threshold):
    """
    Determine which parking spots are occupied based on IoU overlap
    
    Returns: List of spot dictionaries with occupancy info
    """
    results = []
    
    for spot in spots:
        # Convert polygon to bounding box for IoU calculation
        spot_bbox = polygon_to_bbox(spot['coordinates'])
        
        # Check overlap with each detection
        occupied = False
        max_iou = 0.0
        
        for detection in detections:
            iou = compute_iou(spot_bbox, detection['bbox'])
            max_iou = max(max_iou, iou)
            
            if iou >= iou_threshold:
                occupied = True
                # Could break here, but we want to track max_iou
        
        results.append({
            'id': spot['id'],
            'coordinates': spot['coordinates'],
            'occupied': occupied,
            'max_iou': max_iou
        })
    
    return results


def draw_results(image, spots_occupancy, detections):
    """
    Draw parking spots and detections on image
    Green = empty, Red = occupied
    """
    output = image.copy()
    
    # Draw parking spots
    for spot in spots_occupancy:
        color = (0, 0, 255) if spot['occupied'] else (0, 255, 0)  # Red if occupied, green if empty
        
        # Draw polygon
        points = np.array(spot['coordinates'], dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)
        
        # Draw spot ID
        center_x = sum(p[0] for p in spot['coordinates']) // 4
        center_y = sum(p[1] for p in spot['coordinates']) // 4
        cv2.putText(output, f"#{spot['id']}", (center_x-10, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw vehicle detections (blue boxes)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output, f"{det['confidence']:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw occupancy count
    total_spots = len(spots_occupancy)
    occupied_count = sum(1 for s in spots_occupancy if s['occupied'])
    available_count = total_spots - occupied_count
    
    status_text = f"Available: {available_count}/{total_spots}"
    cv2.rectangle(output, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.putText(output, status_text, (20, 38),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return output


def main():
    """Main processing pipeline"""
    
    # Load parking spots
    print("Loading parking spot coordinates...")
    spots = load_parking_spots(SPOTS_JSON)
    print(f"✓ Loaded {len(spots)} parking spots")
    
    # Load YOLO model
    print(f"\nLoading YOLOv8 model ({MODEL_PATH})...")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded")
    
    # Get all frame files
    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    if not frame_files:
        frame_files = sorted(list(FRAMES_DIR.glob("*.jpg")))
    
    if not frame_files:
        print(f"ERROR: No frames found in {FRAMES_DIR}")
        return
    
    print(f"✓ Found {len(frame_files)} frames to process")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    print(f"\nProcessing frames...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print("-" * 60)
    
    for i, frame_path in enumerate(frame_files):
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Warning: Could not load {frame_path}")
            continue
        
        # Detect vehicles
        detections = detect_vehicles(model, image, CONFIDENCE_THRESHOLD)
        
        # Determine occupancy
        spots_occupancy = determine_occupancy(spots, detections, IOU_THRESHOLD)
        
        # Draw results
        output_image = draw_results(image, spots_occupancy, detections)
        
        # Save
        output_path = OUTPUT_DIR / f"detected_{frame_path.name}"
        cv2.imwrite(str(output_path), output_image)
        
        # Print progress
        occupied = sum(1 for s in spots_occupancy if s['occupied'])
        available = len(spots) - occupied
        print(f"Frame {i+1}/{len(frame_files)}: {len(detections)} vehicles detected | Available: {available}/{len(spots)}")
    
    print("-" * 60)
    print(f"\n✓ Processing complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print("  1. Review output images in results/detections/")
    print("  2. Check if detection accuracy looks good")
    print("  3. Adjust thresholds if needed (CONFIDENCE_THRESHOLD, IOU_THRESHOLD)")


if __name__ == "__main__":
    main()