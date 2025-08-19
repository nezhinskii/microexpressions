import cv2
import numpy as np
from ultralytics import YOLO

def detect_faces(image_path, model_path, device="cuda:0"):
    image = cv2.imread(image_path)
    model = YOLO(model_path)
    results = model.predict(image_path, device=device, conf=0.02, iou=0.5)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if hasattr(result, "keypoints") and result.keypoints is not None:
            landmarks = result.keypoints.xy.cpu().numpy()
        else:
            continue

        for box, score, lm in zip(boxes, scores, landmarks):
            detections.append({
                'bbox': box.tolist(),
                'score': float(score),
                'landmarks': lm.tolist()
            })

    return image, detections

def align_face(image, landmarks, bbox, output_size=(112, 112)):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) if dX != 0 else 0

    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    center_x = (x1 + x2) / 2
    
    mean_x = (left_eye[0] + right_eye[0] + nose[0]) / 3
    offset_x = abs(mean_x - center_x)
    offset_x_ratio = offset_x / bbox_width
    max_offset_x_ratio = 0.2
    if offset_x_ratio > max_offset_x_ratio:
        raise ValueError(f"Horizontal offset of keypoints is too large (offset_x_ratio={offset_x_ratio:.2f}), likely profile view")

    mean_eye_y = (left_eye[1] + right_eye[1]) / 2
    offset_y = nose[1] - mean_eye_y
    offset_y_ratio = abs(offset_y) / bbox_height
    min_offset_y_ratio = 0.08
    if offset_y_ratio < min_offset_y_ratio:
        raise ValueError(f"Vertical offset of nose is too small (offset_y_ratio={offset_y_ratio:.2f}), likely top/bottom view")
    
    bb_points = np.array([
        [x1, y1],
        [x2, y1],
        [x1, y2],
        [x2, y2]
    ], dtype=np.float32)
    bb_points_homogeneous = np.hstack([bb_points, np.ones((4, 1))])
    rotated_bb_points = np.dot(bb_points_homogeneous, M.T)[:, :2]
    
    x_min = np.min(rotated_bb_points[:, 0])
    x_max = np.max(rotated_bb_points[:, 0])
    y_min = np.min(rotated_bb_points[:, 1])
    y_max = np.max(rotated_bb_points[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    padding = max(width, height) * 0.1
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)
    
    cropped_face = aligned_image[int(y_min):int(y_max), int(x_min):int(x_max)]
    
    h, w = cropped_face.shape[:2]
    scale = min(output_size[0] / w, output_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_face = cv2.resize(cropped_face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    final_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    x_offset = (output_size[0] - new_w) // 2
    y_offset = (output_size[1] - new_h) // 2
    final_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_face
    
    return final_image