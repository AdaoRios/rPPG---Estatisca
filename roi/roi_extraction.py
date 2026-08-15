import cv2
import numpy as np

def get_roi_mask(landmarks_px, roi_indices, frame_shape, erode_kernel_size=5):
    """Build an eroded ROI mask to eliminate edge motion noise."""
    points = np.array(
        [landmarks_px[idx] for idx in roi_indices if idx < len(landmarks_px)],
        dtype=np.int32
    )
    if len(points) < 3:
        return None

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    cv2.fillPoly(mask, [points], 255)
    
    # Erode edges to prevent boundary mixing during slight head motion
    if erode_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size)
        )
        mask = cv2.erode(mask, kernel, iterations=1)
        
    return mask


def extract_roi_means(rgb_frame, landmarks_px, roi_points):
    """Extract spatial RGB/BGR means for each ROI."""
    frame_means = {}
    
    for roi_name, indices in roi_points.items():
        mask = get_roi_mask(landmarks_px, indices, rgb_frame.shape, erode_kernel_size=5)
        
        if mask is None or cv2.countNonZero(mask) < 50:
            return None
            
        # Get mean RGB values inside the mask
        frame_means[roi_name] = cv2.mean(rgb_frame, mask=mask)[:3]
        
    return frame_means