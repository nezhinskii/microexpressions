from src.demo_helpers import find_with_extension
from src.alignment import align_face, detect_faces
from src.landmarks import run_landmark_model, load_model, get_transforms
from src.procrustes import normalize_points, procrustes_normalization
from src.extract_facial_features_and_filter import extract_features
from src.types_of_faces import get_fragment_features_center
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

def find_farthest_clusters(num_farthest, fragment_features_center, centers_path = r'models\types_of_faces\centers_df.csv'):
    centers_df = pd.read_csv(centers_path)
    id_center_dict = {k : np.array(list(v.values())) for k, v in centers_df.set_index('face_type').to_dict('index').items()}
    # find farthest among themselves
    selected = [fragment_features_center]
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

def augment_fragment(
    original_fragment_pr_lm_df:pd.DataFrame,
    original_fragment_features_df:pd.DataFrame, 
    mean_faces: dict, 
    num_augments: int = 4, 
    smoothing: float = 0.01
):
    fragment_features_center = get_fragment_features_center(original_fragment_features_df)
    farthest_clusters = find_farthest_clusters(num_augments, fragment_features_center)

    def fit_tps(source_points, target_points, smoothing):
        interpolator = RBFInterpolator(source_points, target_points, kernel='thin_plate_spline', smoothing=smoothing)
        return interpolator
    
    def apply_tps(interpolator, input_points):
        return interpolator(input_points)
    
    procrustes_fragment_lm = []
    for _, row in original_fragment_pr_lm_df.iterrows():
        fragment_landmarks = row[[ind for ind in row.index if re.match(r'(n_)?[xy]\d+', ind)]]
        procrustes_fragment_lm.append(fragment_landmarks.to_numpy().reshape(-1, 2).astype(np.float32))
    
    onset_points = procrustes_fragment_lm[0]
    augmented_dict = {}
    for cl_id in farthest_clusters:
        target_points = mean_faces[cl_id]
        interpolator = fit_tps(onset_points, target_points, smoothing)
        augmented_sequence = []
        for frame in procrustes_fragment_lm:
            augmented_frame = apply_tps(interpolator, frame)
            augmented_sequence.append(augmented_frame)
        augmented_dict[cl_id] = augmented_sequence
    return augmented_dict, procrustes_fragment_lm