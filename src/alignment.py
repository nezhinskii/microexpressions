import cv2
import numpy as np
import os
import argparse
import logging
from ultralytics import YOLO
from tqdm import tqdm
import mlflow


def detect_faces(image_path, model, device="cuda:0", conf=0.25, iou=0.7):
    image = cv2.imread(image_path)
    if isinstance(model, str):
        model = YOLO(model)
    results = model.predict(image, device=device, conf=conf, iou=iou, verbose=False)

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

def align_face(image, landmarks, bbox, output_size=(128, 128)):
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


def process_images_celeba(input_path, output_path, model_path, device="cuda:0", output_size=(256, 256), conf=0.25, iou=0.7):
    os.makedirs(output_path, exist_ok=True)
    model = YOLO(model_path)
    
    total_images = 0
    faces_processed = 0
    faces_skipped = 0
    image_files = [f for f in os.listdir(input_path) if f.endswith((".jpg", ".png"))]

    for img_file in tqdm(image_files, desc="Processing images", unit="image"):
        if img_file.endswith((".jpg", ".png")):
            total_images += 1
            img_path = os.path.join(input_path, img_file)
            image, detections = detect_faces(img_path, model, device, conf, iou)
            if not detections:
                logger.info(f"No faces detected in {img_file}, skipping")
                faces_skipped += 1
                continue
            if detections:
                try:
                    aligned_image = align_face(image, np.array(detections[0]["landmarks"]), detections[0]["bbox"], output_size=output_size)
                    output_file = os.path.join(output_path, f"aligned_{img_file}")
                    cv2.imwrite(output_file, aligned_image)
                    logger.info(f"Saved aligned face for {img_file} to {output_file}")
                    faces_processed += 1
                    if faces_processed <= 5:
                        mlflow.log_artifact(output_file)
                except:
                    logger.info(f"Skipping face in {img_file} due to alignment issues")
                    faces_skipped += 1
    mlflow.log_metric("total_images_processed", total_images)
    mlflow.log_metric("faces_processed", faces_processed)
    mlflow.log_metric("faces_skipped", faces_skipped)
                    
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("alignment_errors.log"),
        ]
    )
    
    parser = argparse.ArgumentParser(description="Detect and align faces in images")
    parser.add_argument("--input", required=True, help="Path to input images")
    parser.add_argument("--output", required=True, help="Path to save aligned faces")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--device", default="cuda:0", help="Device to run model on")
    parser.add_argument("--output-size", default="256x256", help="Output size as WIDTHxHEIGHT")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for YOLO")
    args = parser.parse_args()
    output_size = tuple(map(int, args.output_size.split('x')))

    mlflow.set_experiment("face_alignment")
    with mlflow.start_run():
        mlflow.log_param("input_path", args.input)
        mlflow.log_param("output_path", args.output)
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("device", args.device)
        mlflow.log_param("output_size", args.output_size)
        mlflow.log_param("conf", args.conf)
        mlflow.log_param("iou", args.iou)

        process_images_celeba(args.input, args.output, args.model, args.device, output_size, args.conf, args.iou)
