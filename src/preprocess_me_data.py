import os
import pandas as pd
import cv2
import numpy as np
import torch
import re
import onnxruntime as ort
import argparse
import shutil
from tqdm import tqdm
from PIL import Image
from alignment import align_face, detect_faces, create_session
from landmarks import run_landmark_model, load_model, get_transforms
from procrustes import load_meanface, compute_similarity_params, apply_similarity_transform
from extract_facial_features_and_filter import extract_features

def get_sorted_fragment_filenames(fragment_path:str):
    fragment_filenames = os.listdir(fragment_path)
    def max_number_key(s):
        nums = re.findall(r'\d+', str(s))
        return max(map(int, nums), default=float('-inf'))
    fragment_filenames = sorted(fragment_filenames, key=max_number_key)
    return fragment_filenames

def align_fragment_frames(fragment_path, fragment_filenames, detect_model, output_size, aligned_path, input_name=None, save_aligned=False):
    aligned_frames = []
    if isinstance(aligned_path, str) and save_aligned:
        os.makedirs(aligned_path, exist_ok=True)
    for frame_name in fragment_filenames:
        frame_path = os.path.join(fragment_path, frame_name)
        image, detections = detect_faces(frame_path, detect_model, input_name)
        aligned_image = align_face(image, np.array(detections[0]["landmarks"]), detections[0]["bbox"], output_size)
        if isinstance(aligned_path, str) and save_aligned:
            aligned_frame_path = os.path.join(aligned_path, frame_name)
            cv2.imwrite(aligned_frame_path, cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        aligned_frames.append(aligned_image)
    return aligned_frames

def compute_eye_tilt_angle(points, left_eye_slice=slice(36, 42), right_eye_slice=slice(42, 48)):
    left_center = np.mean(points[left_eye_slice], axis=0)
    right_center = np.mean(points[right_eye_slice], axis=0)
    
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    
    return np.arctan2(dy, dx)

def rotate_fragment_to_horizontal(fragment_landmarks, left_eye_slice=slice(36, 42), right_eye_slice=slice(42, 48)):
    N = len(fragment_landmarks)
    
    tilt_angles = np.zeros(N)
    for i in range(N):
        tilt_angles[i] = compute_eye_tilt_angle(
            fragment_landmarks[i],
            left_eye_slice,
            right_eye_slice
        )
    
    mean_sin = np.mean(np.sin(tilt_angles))
    mean_cos = np.mean(np.cos(tilt_angles))
    avg_tilt_angle = np.arctan2(mean_sin, mean_cos)
    
    cos_a = np.cos(avg_tilt_angle)
    sin_a = np.sin(avg_tilt_angle)
    R_horizontal = np.array([[cos_a, -sin_a],
                             [sin_a,  cos_a]])
    
    rotated_fragment = np.zeros_like(fragment_landmarks)
    for i in range(N):
        centroid = np.mean(fragment_landmarks[i], axis=0)
        centered = fragment_landmarks[i] - centroid
        rotated = centered @ R_horizontal + centroid
        rotated_fragment[i] = rotated
    
    return rotated_fragment

def proctrustes_fragment_frames(meanface_path, fragment_landmarks, slice_for_alignment=None, allow_reflection=True):
    meanface = load_meanface(meanface_path)
    rotated_fragment = rotate_fragment_to_horizontal(fragment_landmarks)

    N = len(rotated_fragment)

    scales = np.zeros(N)
    rotations = np.zeros((N, 2, 2))
    translations = np.zeros((N, 2))
    for i in range(N):
        target_points = rotated_fragment[i]
        
        scale, R, t = compute_similarity_params(
            target_points, meanface,
            slice_for_alignment=slice_for_alignment,
            allow_reflection=allow_reflection
        )
        
        scales[i] = scale
        rotations[i] = R
        translations[i] = t
    
    avg_scale = np.mean(scales)

    thetas = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    mean_sin = np.mean(np.sin(thetas))
    mean_cos = np.mean(np.cos(thetas))
    avg_theta = np.arctan2(mean_sin, mean_cos)
    avg_R = np.array([
        [np.cos(avg_theta), -np.sin(avg_theta)],
        [np.sin(avg_theta),  np.cos(avg_theta)]
    ])

    avg_t = np.mean(translations, axis=0)

    normalized_fragment = np.zeros_like(rotated_fragment)

    for i in range(N):
        original_points = rotated_fragment[i]
        normalized_fragment[i] = apply_similarity_transform(
            original_points, avg_scale, avg_R, avg_t
        )

    return normalized_fragment

# def get_face_bbox(landmarks: np.ndarray, width: int, height: int, scale_factor: float = 1.2) -> tuple[list[int], int]:
#     min_x, min_y = np.min(landmarks, axis=0)
#     max_x, max_y = np.max(landmarks, axis=0)
#     center_x = (min_x + max_x) // 2
#     center_y = (min_y + max_y) // 2
#     base_size = int(max(max_x - min_x, max_y - min_y) * scale_factor)
#     size = (base_size // 2 * 2)  # Make even
#     x1 = max(center_x - size // 2, 0)
#     y1 = max(center_y - size // 2, 0)
#     size = min(width - x1, size)
#     size = min(height - y1, size)
#     x2, y2 = x1 + size, y1 + size
#     return [int(x1), int(y1), int(x2), int(y2)], size

def merge_landmarks(detected: list[np.ndarray], forward_pred: list[np.ndarray], backward_pred: list[np.ndarray],
                    prior_variance: np.ndarray, forward_status: np.ndarray = None, backward_status: np.ndarray = None,
                    q_noise: float = 0.6, min_consistency_error: float = 5.0) -> tuple[np.ndarray, list[bool], np.ndarray]:
    num_points = detected[0].shape[0]
    current_detected = detected[1]
    current_forward = forward_pred[1]
    prev_forward = forward_pred[0]
    prev_backward = backward_pred[0]

    backward_diff = prev_backward - prev_forward
    backward_dist = np.linalg.norm(backward_diff, axis=1, keepdims=True)

    reliable = [True] * num_points
    for i in range(num_points):
        if backward_dist[i] > min_consistency_error:  # Bad backward consistency
            reliable[i] = False
        if forward_status is not None and forward_status[i][0] == 0:
            reliable[i] = False
        if backward_status is not None and backward_status[i][0] == 0:
            reliable[i] = False
            
    detected_diff = detected[1] - detected[0]
    detected_dist = np.linalg.norm(detected_diff, axis=1, keepdims=True)
    predicted_diff = forward_pred[1] - forward_pred[0]
    predicted_dist = np.linalg.norm(predicted_diff, axis=1, keepdims=True)
    predicted_dist = np.maximum(predicted_dist, 1e-6)
    measurement_variance = np.maximum(1.0, (detected_dist / predicted_dist) ** 2).reshape(num_points)

    merged = current_detected.copy().astype(float)
    for i in range(num_points):
        if reliable[i]:
            prior_variance[i] += q_noise
            kalman_gain = prior_variance[i] / (prior_variance[i] + measurement_variance[i])
            merged[i] = current_forward[i] + kalman_gain * (current_detected[i] - current_forward[i])
            prior_variance[i] = (1 - kalman_gain) * prior_variance[i]

    return merged, reliable, prior_variance


def calibrate_landmarks(frames: list[np.ndarray], normalized_landmarks: list[np.ndarray], lk_params: dict = None) -> list[np.ndarray]:
    num_frames = len(frames)
    if num_frames != len(normalized_landmarks):
        raise ValueError("Frames and landmarks must match in length.")
    if num_frames == 0:
        return []

    if lk_params is None:
        lk_params = dict(winSize=(15, 15), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Denormalize landmarks to absolute pixel coords for each frame
    absolute_landmarks = []
    for frame, norm_lm in zip(frames, normalized_landmarks):
        h, w = frame.shape[:2]
        abs_lm = np.zeros_like(norm_lm)
        abs_lm[:, 0] = (norm_lm[:, 0] * (w / 2)) + (w / 2)
        abs_lm[:, 1] = (norm_lm[:, 1] * (h / 2)) + (h / 2)
        absolute_landmarks.append(abs_lm.astype(float))

    # Initialize tracking
    tracked_lms = [absolute_landmarks[0].astype(float)]
    num_points = absolute_landmarks[0].shape[0]
    prior_variance = np.ones(num_points, dtype=float) * 2.0

    # Pairwise tracking and merging
    for i in range(num_frames - 1):
        face_pair = frames[i:i + 2]
        lm_pair = absolute_landmarks[i:i + 2]
        start_points = tracked_lms[i].astype(np.float32)
        target_points = lm_pair[1].astype(np.float32)

        # Forward optical flow
        forward_points, forward_status, forward_err = cv2.calcOpticalFlowPyrLK(
            face_pair[0], face_pair[1], start_points, target_points, **lk_params,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )

        # Backward optical flow
        backward_points, backward_status, backward_err = cv2.calcOpticalFlowPyrLK(
            face_pair[1], face_pair[0], forward_points, start_points, **lk_params,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )

        # Round to ints for merge
        forward_pair = [np.rint(tracked_lms[i]).astype(int), np.rint(forward_points).astype(int)]
        backward_pair = [np.rint(backward_points).astype(int), np.rint(forward_points).astype(int)]

        # Merge
        merged_lm, reliable_flags, prior_variance = merge_landmarks(
            lm_pair, forward_pair, backward_pair, prior_variance,
            forward_status, backward_status
        )
        tracked_lms.append(merged_lm)

    calibrated_normalized = []
    original_sizes = [frame.shape[:2] for frame in frames]
    for (h, w), abs_lm in zip(original_sizes, tracked_lms):
        norm_lm = np.zeros_like(abs_lm)
        norm_lm[:, 0] = (abs_lm[:, 0] / (w / 2.0)) - 1.0
        norm_lm[:, 1] = (abs_lm[:, 1] / (h / 2.0)) - 1.0
        calibrated_normalized.append(norm_lm)

    return calibrated_normalized

def process_fragment(
    base_path:str, 
    fragment_data:pd.Series, 
    detect_session:ort.InferenceSession, 
    lm_model_type:str, 
    lm_model, 
    lm_model_transforms, 
    lm_extra_data, 
    meanface_path: str,
    alignment: bool = False,
    aligned_size:tuple[int, int] = (256, 256), 
    device:str = 'cuda:0', 
    fps:int = 30,
):
    subject=fragment_data['Subject']; filename=fragment_data['Filename']; onset=fragment_data['Onset']
    fragment_dir = subject + '_' + filename + '_' + str(onset)
    fragment_path = os.path.join(base_path, fragment_dir)
    fragment_filenames = get_sorted_fragment_filenames(fragment_path)

    # alignment
    aligned_dir = fragment_dir + '_aligned'
    aligned_path = os.path.join(base_path, aligned_dir)
    input_name = detect_session.get_inputs()[0].name
    if alignment:
        frames = align_fragment_frames(fragment_path, fragment_filenames, detect_session, aligned_size, aligned_path, input_name, True)
    else:
        frames = [cv2.cvtColor(cv2.imread(os.path.join(fragment_path, fname)), cv2.COLOR_BGR2RGB) for fname in fragment_filenames]

    # landmark
    if lm_model_transforms is None:
        transformed_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    else:
        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        transformed_frames = torch.stack([lm_model_transforms(frame) for frame in pil_frames])
    fragment_landmarks = run_landmark_model(lm_model_type, lm_model, transformed_frames, lm_extra_data, device)
    # calibration step (FLCM)
    calibrated_landmarks = calibrate_landmarks(frames, fragment_landmarks)
    fragment_landmarks = calibrated_landmarks 
    raw_lm_data = []
    for filename, landmarks in zip(fragment_filenames, fragment_landmarks):
        if landmarks is None:
            raise ValueError(f"No face found on {filename}")
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.flatten()
        result = {'filename': filename}
        for i in range(68):
            result[f'x{i}'] = landmarks[2 * i]
            result[f'y{i}'] = landmarks[2 * i + 1]
        raw_lm_data.append(result)
    raw_lm_df = pd.DataFrame(raw_lm_data)


    # rescale axis wih origin aspect ratio
    corrected_landmarks = fragment_landmarks
    if not alignment and frames:
        h, w = frames[0].shape[:2]
        ar = w / h
        if abs(ar - 1.0) > 1e-6:
            corrected_landmarks = []
            for lm in calibrated_landmarks:
                if lm is None:
                    corrected_landmarks.append(None)
                    continue
                if isinstance(lm, torch.Tensor):
                    lm = lm.cpu().numpy()
                lm_corr = lm.copy()
                if ar > 1:
                    lm_corr[:, 1] *= (1 / ar)
                else:
                    lm_corr[:, 0] *= ar
                corrected_landmarks.append(lm_corr)
    # procrustes
    procrustes_fragment_lm = proctrustes_fragment_frames(meanface_path, fragment_landmarks, slice_for_alignment = slice(17, 68))
    coord_columns = [column for column in raw_lm_df.columns if re.match(r'[xy]\d+', column)]
    norm_coord_columns = ['n_' + column for column in coord_columns]
    pr_lm_data = []
    for filename, pr_landmarks in zip(fragment_filenames, procrustes_fragment_lm):
        normalized_flat = pr_landmarks.reshape(-1)
        pr_lm_data.append([filename] + normalized_flat.tolist())
    pr_lm_df = pd.DataFrame(pr_lm_data, columns=['filename'] + norm_coord_columns)

    # extract features
    feature_names, features = extract_features(pr_lm_df)
    features_df = pd.DataFrame(features, columns = feature_names, index = pr_lm_df.index)
    features_df['filename'] = pr_lm_df['filename']

    return fragment_dir, raw_lm_df, pr_lm_df, features_df

def find_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            return os.path.join(folder_path, file)

def process_dataset(
    base_path:str=r'data\raw\casme3',
    output_base_path:str=r'data\me_landmarks\casme3',
    detect_model_path:str=r'models\yolov6s_face.onnx',
    lm_model_type:str=r'pipnet',
    lm_model_path:str=r'models\pipnet.pth',
    face_detector_path:str=r'models\shape_predictor_68_face_landmarks.dat',
    meanface_path:str=r'models\meanface.txt',
    aligned_size:tuple[int, int]=(256, 256), 
    device:str='cuda:0',
    fps:int=30
):
    detect_session = create_session(detect_model_path)
    lm_model, lm_extra_data = load_model(lm_model_type, lm_model_path, device, face_detector_path, meanface_path)
    lm_model_transforms = get_transforms(lm_model_type)
    labels_path = os.path.join(base_path, 'labels.xlsx')
    casme_df = pd.read_excel(labels_path)

    os.makedirs(output_base_path, exist_ok=True)
    shutil.copy2(labels_path, os.path.join(output_base_path, 'labels.xlsx'))
    bad_fragments_path = os.path.join(base_path, 'bad_fragments.txt')
    if os.path.exists(bad_fragments_path):
        bad_fragments_dest_path = os.path.join(output_base_path, 'bad_fragments.txt')
        shutil.copy2(bad_fragments_path, bad_fragments_dest_path)
    bad_images_path = os.path.join(base_path, 'bad_images.txt')
    if os.path.exists(bad_images_path):
        bad_images_dest_path = os.path.join(output_base_path, 'bad_images.txt')
        shutil.copy2(bad_images_path, bad_images_dest_path)
    
    fragments_path = base_path
    for index, row in tqdm(casme_df.iterrows(), total=len(casme_df)):
        subject=row['Subject']; filename=row['Filename']; onset=row['Onset']; offset=row['Offset']
        if (offset - onset > 2 * fps):
            continue
        fragment_dir = subject + '_' + filename + '_' + str(onset)
        try:
            fragment_dir, raw_lm_df, pr_lm_df, features_df = process_fragment(
                base_path=fragments_path, 
                fragment_data=row, 
                detect_session=detect_session, 
                lm_model_type=lm_model_type, 
                lm_model=lm_model, 
                lm_model_transforms=lm_model_transforms, 
                lm_extra_data=lm_extra_data, 
                meanface_path=meanface_path,
                aligned_size=aligned_size, 
                device=device, 
                fps=fps
            )
            output_fragment_path = os.path.join(output_base_path, fragment_dir)
            os.makedirs(output_fragment_path, exist_ok=True)
            raw_lm_df.to_csv(os.path.join(output_fragment_path, "raw_landmarks.csv"), sep=';')
            pr_lm_df.to_csv(os.path.join(output_fragment_path, "procrustes_lm.csv"), sep=';')
            features_df.to_csv(os.path.join(output_fragment_path, "features.csv"), sep=';')
        except Exception as e:
            print(f"ERROR on fragment {fragment_dir}: {e}.\nSkip...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r'data\raw\casme3',
                        help="Base path to ME dataset")
    parser.add_argument("--output", default=r'data\me_landmarks\casme3',
                        help="Path to save processed fragments")
    parser.add_argument("--detection_model", default=r'models\yolov6s_face.onnx',
                        help="Path to detection model (YOLO face)")
    parser.add_argument("--lm_model_type", default='starnet',
                        choices=["facexformer", "pipnet", "starnet"],
                        help="Landmark model type")
    parser.add_argument("--lm_model", default=r'models\300W_STARLoss_NME_2_87.pkl',
                        help="Path to landmark model")
    parser.add_argument("--face_detector_path", default=r'models\shape_predictor_68_face_landmarks.dat', 
                        help="Path to face detector (for starnet)")
    parser.add_argument("--meanface_path", default=r'models\meanface.txt',
                        help="Path to meanface.txt (for pipnet)")
    parser.add_argument("--aligned_size", default="256x256",
                        help="Output size as WIDTHxHEIGHT (e.g. 256x256)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to run models on (e.g. cuda:0 or cpu)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for processing video fragments")
    args = parser.parse_args()
    
    if isinstance(args.aligned_size, str):
        aligned_size = tuple(map(int, args.aligned_size.split('x')))
    else:
        aligned_size = args.aligned_size
    
    process_dataset(
        base_path=args.input,
        output_base_path=args.output,
        detect_model_path=args.detection_model,
        lm_model_type=args.lm_model_type,
        lm_model_path=args.lm_model,
        face_detector_path=args.face_detector_path,
        meanface_path=args.meanface_path,
        aligned_size=aligned_size,
        device=args.device,
        fps=args.fps
    )