# %%
# content: instance segmentation with tracking and pre-given class label

import cv2

from ultralytics import solutions

# cap = cv2.VideoCapture("data/raw_input/20250417_120203.mp4")
cap = cv2.VideoCapture("data/raw_input/20250418_212412.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("data/processed_output/instance_segmentation_with_tracking_laptop.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize instance segmentation object
isegment = solutions.InstanceSegmentation(
    show=True,  # display the output
    model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
    classes=[63],  # segment specific classes i.e, person and car with pretrained model.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = isegment(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows