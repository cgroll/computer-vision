# %%
# content: instance segmentation with text prompt
# conclusion: takes pretty long and does not work very well

from ultralytics import FastSAM
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

# Define an inference source
source = "data/raw_input/person_in_room.mp4"

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# %%

# Process video frame by frame
cap = cv2.VideoCapture(source)
all_results = []
error_count = 0

# Get total frame count for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_num in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break
        
    try:
        # Resize frame to 1024x1024
        frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)

        # Run inference with texts prompt
        results = model(frame, texts="person", imgsz=1024)
        all_results.append(results)
        
    except Exception as e:
        error_count += 1
        print(f"Error processing frame {frame_num}: {error_count}")
        # Append None to maintain frame alignment
        all_results.append(None)

cap.release()
print(f"Total errors encountered: {error_count}")


# %%

# Create video writer
cap = cv2.VideoCapture(source)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create output video writer
output_path = "data/processed_output/person_segmentation.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))

# Process each frame and write to video
for frame_num, frame_results in enumerate(tqdm(all_results, desc="Creating video")):

    # Get frame from video
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    # Resize frame to 1024x1024
    frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)

    # Get masks from results
    if frame_results is None:
        masks = None
    else:
        masks = frame_results[0].masks.data
    
    # Create empty overlay for masks
    overlay = np.zeros((1024, 1024, 4), dtype=np.uint8)
    
    if masks is not None:
        # Sort masks by area (largest first)
        areas = np.sum(masks.cpu().numpy(), axis=(1,2))
        sorted_indices = np.argsort(areas)[::-1]
        
        # Apply each mask with random color
        for idx in sorted_indices:
            mask = masks[idx].cpu().numpy()
            
            # Generate random color with alpha
            color = np.random.randint(0, 255, 3)
            alpha = 0.6
            
            # Create RGBA color
            rgba = np.append(color, int(255 * alpha))
            
            # Apply color where mask is True
            overlay[mask > 0] = rgba
            
            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0,0,255,230), 2)

    # Blend original frame with overlay
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    blended = cv2.addWeighted(frame_rgba, 1, overlay, 0.5, 0)
    
    # Convert back to BGR for video writing
    processed_frame = cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR)
    
    # Write frame to video
    out.write(processed_frame)

# Release video writer
out.release()
print(f"Video saved to {output_path}")

# %%
