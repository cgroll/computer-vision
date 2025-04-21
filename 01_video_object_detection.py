# %%
# content: object detection on video, showing bounding boxes and class labels

from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('yolo11n.pt')

# Path to video file
video_path = 'data/raw_input/person_in_room.mp4'

# %%

# Run inference on video
results = model(video_path, stream=True)  # Generator of Results objects

# Get video properties from first frame
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

# Create video writer
output_path = 'data/processed_output/person_in_room_object_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process results frame by frame
for result in results:
    # Visualize the results on the frame
    annotated_frame = result.plot()
    
    # Write frame to video file
    out.write(annotated_frame)

# Release resources
out.release()

# %%
