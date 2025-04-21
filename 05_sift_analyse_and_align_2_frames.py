# %%
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

video_path = 'data/raw_input/laptop_home_office_daylight.mp4'

def show_blended_frames(frame1, frame2, transformed_corners=None):

    alpha = 0.5

    # Create new blended visualization with aligned frames
    blended = cv2.addWeighted(frame1, alpha, frame2, alpha, 0)

    if transformed_corners is not None:
        # Draw border around warped frame1
        for i in range(transformed_corners.shape[1]-1, -1, -1):
            cv2.line(blended, tuple(transformed_corners[0, i, :]), 
                    tuple(transformed_corners[0, i-1, :]), (0,0,255), thickness=2)

    # Show aligned blended result
    cv2.imshow('Aligned Blend', blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_keypoints_on_frames(frame1, frame2, kp1, kp2):

    # Draw keypoints on both frames
    frame1_kp = cv2.drawKeypoints(frame1, kp1, None, color=(0,255,0), 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame2_kp = cv2.drawKeypoints(frame2, kp2, None, color=(0,255,0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    ax1.imshow(cv2.cvtColor(frame1_kp, cv2.COLOR_BGR2RGB))
    ax1.set_title('Frame 1 with SIFT Keypoints')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(frame2_kp, cv2.COLOR_BGR2RGB))
    ax2.set_title('Frame 2 with SIFT Keypoints') 
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def draw_matches(frame1, frame2, kp1, kp2, good_matches):

    # Draw matches using matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Draw matches on frame 1
    ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    for match in good_matches[:50]:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        ax1.plot(p1[0], p1[1], 'go')
        # Draw line between matched points
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.5)
    ax1.set_title('Frame 1 Matched Keypoints with Match Lines')
    ax1.axis('off')

    # Draw matches on frame 2  
    ax2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    for match in good_matches[:50]:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        ax2.plot(p2[0], p2[1], 'go')
        # Draw line between matched points
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.5)
    ax2.set_title('Frame 2 Matched Keypoints with Match Lines')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def get_frame_corners(frame):
    corner_0 = np.array([0, 0])
    corner_1 = np.array([frame.shape[1], 0]) 
    corner_2 = np.array([frame.shape[1], frame.shape[0]])
    corner_3 = np.array([0, frame.shape[0]])
    corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
    return corners

def nearest_neighbor_matches_and_homography(des1, des2, kp1, kp2):

    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp2[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    # Find homography matrix
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    return H, good_matches

def compute_ssim(frame1, frame2, mask=None):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between aligned frames
    ssim_score, ssim_map = ssim(frame1_gray, frame2_gray,
                    data_range=frame2_gray.max() - frame2_gray.min(),
                    full=True)
    
    if mask is not None:
        non_overlap_mask = ~mask
        ssim_map_overlap = ssim_map.copy()
        ssim_map_overlap[non_overlap_mask] = np.nan

        avg_ssim_overlap = np.nanmean(ssim_map_overlap)
        ssim_score = avg_ssim_overlap

    return ssim_score, ssim_map

def visualize_ssim(frame1, frame2, mask=None):
    ssim_score, ssim_map = compute_ssim(frame1, frame2, mask)

    # Visualize SSIM map
    plt.figure(figsize=(8,6))
    plt.imshow(ssim_map, cmap='jet')
    plt.colorbar(label='SSIM')
    if mask is not None:
        plt.title(f'SSIM score on overlapping region: {ssim_score:.4f}')
    else:
        plt.title(f'SSIM score: {ssim_score:.4f}')
    plt.show()

# %%

# Load video
cap = cv2.VideoCapture(video_path)
# Frame numbers to extract
reference_frame_num = 0
frame_to_align_num = 30

cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num)
ret, ref_frame = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_align_num)
ret, frame_to_align = cap.read()

# Release video capture
cap.release()

# %%

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both frames
kp1, des1 = sift.detectAndCompute(ref_frame, None)
kp2, des2 = sift.detectAndCompute(frame_to_align, None)

# %%

H, good_matches = nearest_neighbor_matches_and_homography(des1, des2, kp1, kp2)

# %%

draw_keypoints_on_frames(ref_frame, frame_to_align, kp2, kp1)

# %%

draw_matches(ref_frame, frame_to_align, kp2, kp1, good_matches)

# %% apply homography

height, width = ref_frame.shape[:2]
aligned_frame = cv2.warpPerspective(frame_to_align, H, (width, height))

corners = get_frame_corners(frame_to_align)

transformed_corners = cv2.perspectiveTransform(corners, H)
transformed_corners = np.array(transformed_corners, dtype=np.int32)

# Create mask for non-overlapping regions
mask = np.zeros_like(cv2.cvtColor(frame_to_align, cv2.COLOR_BGR2GRAY))
cv2.fillPoly(mask, [transformed_corners], 255)
mask = mask.astype(bool)

# %%

show_blended_frames(ref_frame, frame_to_align)

# %%

show_blended_frames(ref_frame, aligned_frame, transformed_corners)

# %%

# visualize_ssim(ref_frame, frame_to_align)
# visualize_ssim(ref_frame, aligned_frame)
visualize_ssim(ref_frame, aligned_frame, mask)


# %%

# keep track of:
# - H, mask, aligned_frame
# - ssim score

# %%

# Get video capture object
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames in video: {total_frames}")

# Lists to store results
homography_matrices = []
aligned_frames = []
ssim_scores = []

# Read first frame as reference
ret, ref_frame = cap.read()
if not ret:
    print("Could not read first frame")
    cap.release()
    exit()

# Iterate through remaining frames
frame_count = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    print(f"Processing frame {frame_count}/{total_frames}")
    
    # Get keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_frame, None)
    kp2, des2 = sift.detectAndCompute(frame, None)
    
    # Get homography matrix and matches
    H, good_matches = nearest_neighbor_matches_and_homography(des1, des2, kp1, kp2)
    
    # Align frame
    height, width = ref_frame.shape[:2]
    aligned_frame = cv2.warpPerspective(frame, H, (width, height))
    
    # Get corners and mask
    corners = get_frame_corners(frame)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    transformed_corners = np.array(transformed_corners, dtype=np.int32)
    
    mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv2.fillPoly(mask, [transformed_corners], 255)
    mask = mask.astype(bool)
    
    # Compute SSIM score
    score = ssim(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2GRAY),
                 full=True)[0]
    
    # Store results
    homography_matrices.append(H)
    aligned_frames.append(aligned_frame)
    ssim_scores.append(score)
    
    frame_count += 1

# Release video capture
cap.release()

print("Processing complete")
print(f"Average SSIM score: {np.mean(ssim_scores):.3f}")

# %%

# Create DataFrame with SSIM scores
import pandas as pd

df_scores = pd.DataFrame({
    'Frame': range(1, len(ssim_scores) + 1),
    'SSIM Score': ssim_scores
})

# Plot SSIM scores
plt.figure(figsize=(12, 6))
plt.plot(df_scores['Frame'], df_scores['SSIM Score'], '-b')
plt.title('SSIM Scores Across Video Frames')
plt.xlabel('Frame Number') 
plt.ylabel('SSIM Score')
plt.grid(True)
plt.show()

# %%

# Create video of aligned frames
output_path = Path('data/processed_output/sift_aligned_laptop_daylight.mp4')
output_path.parent.mkdir(parents=True, exist_ok=True)

# Get dimensions from first frame
height, width = aligned_frames[0].shape[:2]

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

# Write frames to video
for frame in aligned_frames:
    out.write(frame)

# Release video writer
out.release()

print(f"Aligned video saved to: {output_path}")

# %% select region to track

points = [(606, 325), (1245, 327), (1246, 707), (606, 706)]

# %%

# Convert points to numpy array format
points_array = np.array(points, dtype=np.int32)

# Make a copy of first frame to draw on
frame_with_polygon = ref_frame.copy()

# Draw filled polygon with some transparency
overlay = frame_with_polygon.copy()
cv2.fillPoly(overlay, [points_array], (0, 255, 0))  # Green fill
frame_with_polygon = cv2.addWeighted(overlay, 0.3, frame_with_polygon, 0.7, 0)

# Draw polygon outline
cv2.polylines(frame_with_polygon, [points_array], True, (0, 255, 0), 2)  # Green outline

# Display the frame with polygon
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(frame_with_polygon, cv2.COLOR_BGR2RGB))
plt.title('Selected Region to Track')
plt.axis('off')
plt.show()

# %%

# Create video writer for tracked region visualization
output_path = Path('data/processed_output/sift_tracked_laptop_daylight.mp4')
output_path.parent.mkdir(parents=True, exist_ok=True)

# Open video capture
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # Skip to second frame (index 1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# Loop through all frames
for frame_num in range(frame_count-1):
    # Get homography matrix for this frame
    H = homography_matrices[frame_num]
    
    # Transform points using inverse homography matrix
    transformed_points = cv2.perspectiveTransform(points_array.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(H))
    transformed_points = transformed_points.astype(np.int32)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Draw filled polygon with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [transformed_points], (0, 255, 0))  # Green fill
    frame_with_overlay = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw polygon outline
    cv2.polylines(frame_with_overlay, [transformed_points], True, (0, 255, 0), 2)  # Green outline
    
    # Write frame to video
    out.write(frame_with_overlay)

# Clean up
cap.release()
out.release()

print(f"Tracked region video saved to: {output_path}")

# %%

