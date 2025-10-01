from src.demo_helpers import find_with_extension
from src.alignment import align_face, detect_faces
from src.landmarks import run_landmark_model, load_model, get_transforms
from src.procrustes import normalize_points, procrustes_normalization
from src.extract_facial_features_and_filter import extract_features
from src.types_of_faces import get_cells_for_new_points
from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import re
from PIL import Image
from scipy.interpolate import RBFInterpolator

def align_fragment_frames(fragment_path, detect_model, device, output_size, aligned_path, save_aligned=False):
    aligned_frames = []
    if isinstance(aligned_path, str) and save_aligned and not os.path.exists(aligned_path):
        os.makedirs(aligned_path)
    for frame_name in os.listdir(fragment_path):
        frame_path = os.path.join(fragment_path, frame_name)
        image, detections = detect_faces(frame_path, detect_model, device)
        aligned_image = align_face(image, np.array(detections[0]["landmarks"]), detections[0]["bbox"], output_size)
        aligned_frame_path = os.path.join(aligned_path, frame_name)
        if isinstance(aligned_path, str) and save_aligned:
            cv2.imwrite(aligned_frame_path, aligned_image)
        aligned_frames.append(aligned_image)
    return aligned_frames

def proctrustes_fragment_frames(meanface_path, fragment_landmarks):
    with open(meanface_path) as f:
        meanface = f.readlines()[0]
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    normalized_meanface, _, _ = normalize_points(meanface)

    onset_lm = fragment_landmarks[0]
    _, fixed_centroid, fixed_scale, fixed_R = procrustes_normalization(onset_lm.numpy(), normalized_meanface)

    def apply_fixed_procrustes(frame_points, fixed_centroid, fixed_scale, fixed_R):
        centered = frame_points - fixed_centroid
        scaled = centered / fixed_scale
        normalized = scaled.dot(fixed_R)
        return normalized
    
    procrustes_fragment_lm = [
        apply_fixed_procrustes(frame_lm.numpy(), fixed_centroid, fixed_scale, fixed_R) 
        for frame_lm in fragment_landmarks
        ]
    return procrustes_fragment_lm

def find_farthest_clusters(num_farthest, fragment_clusters, centers_path = r'models\types_of_faces\centers_df.csv'):
    centers_df = pd.read_csv(centers_path)
    # weighted fragment coords
    id_center_dict = {k : np.array(list(v.values())) for k, v in centers_df.set_index('face_type').to_dict('index').items()}
    sum_frames = np.zeros_like(id_center_dict[0])
    for i in fragment_clusters:
        sum_frames += id_center_dict[i]
    fragment_coords = sum_frames / len(fragment_clusters)

    # find farthest among themselves
    selected = [fragment_coords]
    selected_indices = []  
    for _ in range(num_farthest):
        max_dist = -1
        max_dist_idx = -1
        for id, center in id_center_dict.items():
            dist_sum = np.sum(np.linalg.norm(center - selected_point) for selected_point in selected)
            if dist_sum > max_dist:
                max_dist = dist_sum
                max_dist_idx = id
        selected.append(id_center_dict[max_dist_idx])
        selected_indices.append(max_dist_idx)
        del id_center_dict[max_dist_idx]
    return selected_indices

def augment_fragment(base_path, fragment_data, cl_faces, detect_model, device, aligned_size, 
                     lm_model_type, lm_model, lm_model_transforms, lm_extra_data, meanface_path, num_farthest):
    subject=fragment_data['Subject']; filename=fragment_data['Filename']; onset=fragment_data['Onset']; apex=fragment_data['Apex'] 
    offset=fragment_data['Offset']; au_str=fragment_data['AU']; objective_class=fragment_data['Objective class']

    # alignment
    fragment_dir = subject + '_' + filename + '_' + str(onset)
    aligned_dir = fragment_dir + '_aligned'
    fragment_path = os.path.join(base_path, fragment_dir)
    aligned_path = os.path.join(base_path, aligned_dir)
    aligned_frames = align_fragment_frames(fragment_path, detect_model, device, aligned_size, aligned_path, True)

    # landmark
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in aligned_frames]
    transformed_frames = torch.stack([lm_model_transforms(frame) for frame in pil_frames])
    fragment_landmarks = run_landmark_model(lm_model_type, lm_model, transformed_frames, lm_extra_data, device)

    # procrustes
    procrustes_fragment_lm = proctrustes_fragment_frames(meanface_path, fragment_landmarks)
    
    # extract features
    lm_columns = []
    for i in range(procrustes_fragment_lm[0].shape[0]):
        lm_columns.append(f'n_x{i}')
        lm_columns.append(f'n_y{i}')
    frames_features = []
    for frame_lm in procrustes_fragment_lm:
        lined_frame_lm = frame_lm.reshape(1,-1)
        frame_df = pd.DataFrame(lined_frame_lm, columns=lm_columns)
        feature_names, frame_features = extract_features(frame_df)
        frames_features.append(frame_features)
    # clusters
    frames_features_df = pd.DataFrame(np.vstack(frames_features), columns=feature_names)
    cells = get_cells_for_new_points(frames_features_df)  

    farthest_clusters = find_farthest_clusters(num_farthest, cells)

    def fit_tps(source_points, target_points, smoothing=0.1):
        interpolator = RBFInterpolator(source_points, target_points, kernel='thin_plate_spline', smoothing=smoothing)
        return interpolator
    def apply_tps(interpolator, input_points):
        return interpolator(input_points)
    onset_points = procrustes_fragment_lm[0]
    augmented_dict = {}
    for cl_id in farthest_clusters:
        target_points = cl_faces[cl_id]
        interpolator = fit_tps(onset_points, target_points, smoothing=0.001)
        augmented_sequence = []
        for frame in procrustes_fragment_lm:
            augmented_frame = apply_tps(interpolator, frame)
            augmented_sequence.append(augmented_frame)
        augmented_dict[cl_id] = augmented_sequence
    return augmented_dict, procrustes_fragment_lm

def prepare_mean_faces(typed_faces_df, procrustes_lm_df):
    merged_df = typed_faces_df.merge(procrustes_lm_df, on=['filename'])
    mean_faces = {}
    for type_id in merged_df['face_type'].unique():
        if type_id == -1:
            continue
        current_type_faces = merged_df[merged_df['face_type'] == type_id]
        coord_columns = [column for column in current_type_faces.columns if re.match(r'(n_)?[xy]\d+', column)]
        mean_faces[type_id] = current_type_faces[coord_columns].mean().to_numpy().reshape(-1, 2)
    return mean_faces