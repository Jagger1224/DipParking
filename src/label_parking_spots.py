import cv2
import json
from pathlib import Path

# Configuration
IMAGE_PATH = Path("data/frames/frame_0000.png")
OUTPUT_JSON = Path("data/parking_spot_coordinates.json")
NUM_SPOTS = 18

# Variables for overall state
spots = {}  # Will store: {1: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], 2: [...], ...}
current_spot_id = 1
current_points = []  # Points for the spot we're currently clicking

# Load image (parking_lot_base.png)
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
display_img = img.copy()

def mouse_callback(event, x, y, flags, param):
    """
    This handles mouse clicks response to add corner points (event driven)
    
    """
    global current_points, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only accept clicks if we haven't finished this spot yet
        if len(current_points) < 4:
            current_points.append((x, y))
            print(f"Spot {current_spot_id} - Point {len(current_points)}/4: ({x}, {y})")
            
            # Redraw
            display_img = img.copy()
            draw_all_spots()
            draw_current_points()

def draw_current_points():
    """
    Draw the points for the spot currently being defined
    
    """
    global display_img
    
    # Draw points
    for i, (x, y) in enumerate(current_points):
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)  # Red dot on selected point (corner)
        cv2.putText(display_img, str(i+1), (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw lines between points (yellow)
    if len(current_points) > 1:
        for i in range(len(current_points)-1):
            cv2.line(display_img, current_points[i], current_points[i+1], (0, 255, 255), 2)
        
        # Close the polygon if we have all 4 points
        if len(current_points) == 4:
            cv2.line(display_img, current_points[3], current_points[0], (0, 255, 255), 2)

def draw_all_spots():
    """
    Draw all completed spots in green
    
    """
    global display_img
    
    for spot_id, points in spots.items():
        # Draw polygon
        pts = [(int(x), int(y)) for x, y in points]
        for i in range(4):
            cv2.line(display_img, pts[i], pts[(i+1)%4], (0, 255, 0), 2)  # Green
        
        # Draw spot ID in center
        center_x = sum(p[0] for p in pts) // 4
        center_y = sum(p[1] for p in pts) // 4
        cv2.putText(display_img, f"#{spot_id}", (center_x-10, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def save_spots():
    """
    Save all spots to JSON file
    
    """
    output_data = {}
    for spot_id, points in spots.items():
        output_data[f"spot_{spot_id}"] = {
            "id": spot_id,
            "coordinates": points
        }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(spots)} spots to {OUTPUT_JSON}")

# Main loop
print(f"Labeling {NUM_SPOTS} parking spots")
print("Click 4 corners for each spot (clockwise from top-left)")
print("Press 'n' after 4 points to go to next spot")
print("Press 's' to save and exit")
print("Press ESC to exit without saving")
print("-" * 50)

cv2.namedWindow("Parking Lot Labeler")
cv2.setMouseCallback("Parking Lot Labeler", mouse_callback)

while True:
    # Add status text
    status_img = display_img.copy()
    status = f"Spot: {current_spot_id}/{NUM_SPOTS} | Points: {len(current_points)}/4 | Completed: {len(spots)}"
    cv2.putText(status_img, status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(status_img, "n=Next | s=Save | ESC=Exit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.imshow("Parking Lot Labeler", status_img)
    key = cv2.waitKey(1) & 0xFF
    
    # ESC - Exit
    if key == 27:  # 27 is ESC
        print("\nExiting without saving")
        break
    
    # 'n' - Next spot (save current and move on automatically when box is drawn)
    elif key == ord('n'):
        if len(current_points) == 4:
            spots[current_spot_id] = current_points.copy()
            print(f"✓ Spot {current_spot_id} saved")
            
            current_spot_id += 1
            current_points = []
            display_img = img.copy()
            draw_all_spots()
            
            if current_spot_id > NUM_SPOTS:
                print(f"\nAll {NUM_SPOTS} spots labeled! Press 's' to save.")
        else:
            print(f"Need 4 points (have {len(current_points)})")
    
    # 's' - Save
    elif key == ord('s'):
        if len(spots) > 0:
            save_spots()
            break
        else:
            print("No spots to save")

# Clear every window
cv2.destroyAllWindows()