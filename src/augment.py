import os
import shutil
import re
import joblib
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from filter_and_cluster_faces import predict_cluster_for_new_face
from preprocess_faces import process_embeddings
from scipy.signal import savgol_filter

def get_fragment_cluster(
    fragment_images_base_path:str,
    original_fragment_pr_lm_df:pd.DataFrame,
    cluster_type:str,
    clusterer_path:str,
    reducer_path:str
):
    fragment_landmarks = []
    coord_columns = [column for column in original_fragment_pr_lm_df.columns if re.match(r'(n_)?[xy]\d+', column)]
    for _, row in original_fragment_pr_lm_df.iterrows():
        frame_landmarks = row[coord_columns]
        fragment_landmarks.append(frame_landmarks.to_numpy().astype(np.float32))
    fragment_filenames = original_fragment_pr_lm_df['filename']
    
    clusterer = joblib.load(clusterer_path)
    reducer = None
    if cluster_type == 'landmarks':
        if reducer_path is None:
            raise ValueError("reducer_path required for landmarks")
        reducer = joblib.load(reducer_path)
        features = fragment_landmarks
    else:
        batch_size = 32
        fragment_embeddings = []
        for i in range(0, len(fragment_filenames), batch_size):
            batch_filenames = fragment_filenames[i:i + batch_size]
            batch_images = np.array([cv2.imread(os.path.join(fragment_images_base_path, filename)) for filename in batch_filenames])
            batch_embeddings = process_embeddings(batch_images)
            fragment_embeddings.extend(batch_embeddings)
        fragment_embeddings = [emb for emb in fragment_embeddings if emb is not None]
    features = np.array(features)
    fragment_clusters = predict_cluster_for_new_face(
        new_landmarks_or_embedding=features,
        clusterer=clusterer,
        cluster_type=cluster_type,
        reducer=reducer,
    )
    values, counts = np.unique(fragment_clusters, return_counts=True)
    most_frequent = values[counts.argmax()]
    return most_frequent

def prepare_cluster_faces(faces_clusters_path):
    df = pd.read_hdf(faces_clusters_path)
    cluster_faces = {}
    columns = [column for column in df.columns if re.match(r'(n_)?[xy]\d+', column)] + ['filename']
    for cluster_id in df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        current_type_faces = df[df['cluster_id'] == cluster_id]
        cluster_faces[cluster_id] = [row for _, row in current_type_faces[columns].iterrows()]
    return cluster_faces

def calculate_ear(eye_points):
    # eye_points: 6 pts per eye (e.g., left: 36-41)
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    vert1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vert2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horiz = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)

def get_neutral_idx(fragment_landmarks):
    fragment_last_idx = len(fragment_landmarks) - 1
    
    first_lm = fragment_landmarks[0]
    last_lm = fragment_landmarks[fragment_last_idx]
    left_eye_pts_first = first_lm[36:42]
    right_eye_pts_first = first_lm[42:48]
    ear_first = (calculate_ear(left_eye_pts_first) + calculate_ear(right_eye_pts_first)) / 2.0
    left_eye_pts_last = last_lm[36:42]
    right_eye_pts_last = last_lm[42:48]
    ear_last = (calculate_ear(left_eye_pts_last) + calculate_ear(right_eye_pts_last)) / 2.0
    
    neutral_idx = 0 if ear_first >= ear_last else fragment_last_idx
    return neutral_idx

regions = {
    'contour': slice(0, 17),
    'brows': slice(17, 27),
    'nose': slice(27, 36),
    'eyes': slice(36, 48),
    'mouth': slice(48, 68),
}

smoothing_ranges = {
    'eyes': (0.01, 0.07),
    'brows': (0.001, 0.05),
    'mouth': (0.001, 0.05),
    'contour': (0.001, 0.05),
    'nose': (0.001, 0.05),
}

amplitude_ranges = {
    'eyes': (0.9, 1.1),
    'brows': (0.6, 1.4),
    'mouth': (0.7, 1.3),
    'contour': (0.8, 1.2),
    'nose': (0.6, 1.4),
}

def apply_temporal_aug(sequence, important_frames, aug_type, n, original_len):
    seq = sequence.copy()
    if aug_type == 'drop':
        indices_to_keep = [i for i in range(original_len) if (i % n != 0) or (i in important_frames)]
        seq = [seq[i] for i in indices_to_keep]
    elif aug_type == 'insert':
        new_seq = []
        for i in range(original_len):
            new_seq.append(seq[i])
            if (i + 1) % n == 0 and i < original_len - 1:
                interp_frame = 0.5 * (seq[i] + seq[i+1])
                new_seq.append(interp_frame)
        seq = new_seq
    return seq

def augment_zone_sequence(
    zone_sequences: dict,
    target_face_points: np.ndarray,
    neutral_idx: int,
    neutral_src: np.ndarray,
    important_frames: list
):
    aug_neutral = target_face_points.copy()
    num_frames = len(next(iter(zone_sequences.values())))
    augmented_sequence = [None] * num_frames
    augmented_sequence[neutral_idx] = aug_neutral
    
    zone_smoothings = {reg: np.random.uniform(low, high) for reg, (low, high) in smoothing_ranges.items()}
    zone_amplitudes = {reg: np.random.uniform(low, high) for reg, (low, high) in amplitude_ranges.items()}
    
    for reg_name, reg_slice in regions.items():
        zone_seq = zone_sequences[reg_name]  # list of arrays for this zone per frame
        neutral_src_zone = neutral_src[reg_slice]
        aug_neutral_zone = aug_neutral[reg_slice]
        current_smoothing = zone_smoothings[reg_name]
        
        for i in range(num_frames):
            if i == neutral_idx:
                continue
            curr_zone = zone_seq[i]
            delta_mapper = RBFInterpolator(
                neutral_src_zone,
                curr_zone,
                kernel='thin_plate_spline',
                smoothing=current_smoothing
            )
            aug_frame_zone = delta_mapper(aug_neutral_zone)
            
            # Apply amplitude scaling
            factor = zone_amplitudes[reg_name]
            delta = aug_frame_zone - aug_neutral_zone
            scaled_delta = factor * delta
            aug_frame_zone = aug_neutral_zone + scaled_delta
            
            if augmented_sequence[i] is None:
                augmented_sequence[i] = np.zeros_like(aug_neutral)
            augmented_sequence[i][reg_slice] = aug_frame_zone
            
    if np.random.rand() < 0.7:
        aug_type = random.choice(['drop', 'insert'])
        n = random.randint(4, 7)
        if aug_type == 'drop':
            estimated_remaining = num_frames - (num_frames // n)
            if estimated_remaining < 6:
                aug_type = 'insert'
        augmented_sequence = apply_temporal_aug(augmented_sequence, important_frames, aug_type, n, num_frames)
    
    # window_length = 5
    # if num_frames > window_length:
    #     aug_array = np.array(augmented_sequence)
    #     for pt in range(68):
    #         for dim in range(2):
    #             aug_array[:, pt, dim] = savgol_filter(aug_array[:, pt, dim], window_length=window_length, polyorder=2)
    #     augmented_sequence = [fr for fr in aug_array]
    
    return augmented_sequence

def augment_fragment(
    original_fragment_pr_lm_df:pd.DataFrame,
    fragment_images_base_path:str,
    cluster_faces:dict,
    cluster_type:str,
    clusterer_path:str,
    fragment_apex:int,
    reducer_path:str,
    num_augments:int=4, 
):
    fragment_cluster = get_fragment_cluster(
        fragment_images_base_path=fragment_images_base_path,
        original_fragment_pr_lm_df=original_fragment_pr_lm_df,
        cluster_type=cluster_type,
        clusterer_path=clusterer_path,
        reducer_path=reducer_path,
    )
    other_clusters = list(cluster_faces.keys())
    other_clusters.remove(fragment_cluster)
    target_clusters = random.sample(other_clusters, num_augments)
    
    fragment_landmarks = []
    coord_columns = [column for column in original_fragment_pr_lm_df.columns if re.match(r'(n_)?[xy]\d+', column)]
    for _, row in original_fragment_pr_lm_df.iterrows():
        frame_landmarks = row[coord_columns]
        fragment_landmarks.append(frame_landmarks.to_numpy().astype(np.float32).reshape(-1, 2))
        
    neutral_idx = get_neutral_idx(fragment_landmarks)
    neutral_src = fragment_landmarks[neutral_idx]
    
    zone_sequences = {reg_name: [] for reg_name in regions}
    for lm in fragment_landmarks:
        for reg_name, reg_slice in regions.items():
            zone_sequences[reg_name].append(lm[reg_slice])

    augmented_dict = {}
    augmented_dict_filenames = {}
    for cl_id in target_clusters:
        target_cluster_faces = cluster_faces[cl_id]
        target_face_row = random.choice(target_cluster_faces)
        target_face_filename = target_face_row['filename']
        target_face_points = target_face_row[coord_columns].to_numpy().astype(np.float32).reshape(-1, 2)
        important_frames = [neutral_idx]
        if not np.isnan(fragment_apex):
            important_frames.append(fragment_apex)
        
        augmented_sequence = augment_zone_sequence(
            zone_sequences=zone_sequences,
            target_face_points=target_face_points,
            neutral_idx=neutral_idx,
            neutral_src=neutral_src,
            important_frames=important_frames
        )
        
        augmented_dict[cl_id] = augmented_sequence
        augmented_dict_filenames[cl_id] = target_face_filename
    return augmented_dict, augmented_dict_filenames

def augment_dataset(
    me_lm_base_path:str=r'data\me_landmarks\casme3_spotting',
    output_base_path:str=r'data\augmented\casme3_spotting',
    me_anno_path:str=r'data\me_landmarks\casme3_spotting\labels.xlsx',
    me_images_base_path:str=r'data\raw\casme3_spotting',
    cluster_type:str='landmarks',
    clusterer_path:str=r'models\face_clustering\cluster_model_landmarks_kmeans.pkl',
    reducer_path:str=r'models\face_clustering\reducer_model_landmarks.pkl',
    faces_clusters_path:str=r'data\processed_faces\cl_celeba_hq_landmarks_kmeans.h5',
    num_augments:int=4,
):
    me_df = pd.read_excel(me_anno_path)
    anno_filename = os.path.basename(me_anno_path)
    anno_dest_path = os.path.join(output_base_path, anno_filename)
    os.makedirs(output_base_path, exist_ok=True)
    shutil.copy2(me_anno_path, anno_dest_path)
    bad_fragments_path = os.path.join(me_lm_base_path, 'bad_fragments.txt')
    if os.path.exists(bad_fragments_path):
        bad_fragments_dest_path = os.path.join(output_base_path, 'bad_fragments.txt')
        shutil.copy2(bad_fragments_path, bad_fragments_dest_path)
    bad_images_path = os.path.join(me_lm_base_path, 'bad_images.txt')
    if os.path.exists(bad_images_path):
        bad_images_dest_path = os.path.join(output_base_path, 'bad_images.txt')
        shutil.copy2(bad_images_path, bad_images_dest_path)
        
    cluster_faces = prepare_cluster_faces(faces_clusters_path)

    fragment_dirs = set(os.listdir(me_lm_base_path))
    has_apex_col = 'Apex' in me_df.columns
    for index, row in tqdm(me_df.iterrows(), total=len(me_df)):
        subject=row['Subject']; filename=row['Filename']; onset=row['Onset']; offset=row['Offset']
        if has_apex_col:
            apex = row['Apex']
        else:
            apex = np.nan
        fragment_dir = subject + '_' + filename + '_' + str(onset)
        if fragment_dir not in fragment_dirs:
            continue
        fragment_path = os.path.join(me_lm_base_path, fragment_dir)
        fragment_images_path = os.path.join(me_images_base_path, fragment_dir)

        original_fragment_pr_lm_path = os.path.join(fragment_path, 'procrustes_lm.csv')
        original_fragment_pr_lm_df = pd.read_csv(original_fragment_pr_lm_path, sep=';', index_col=0)
        original_fragment_features_path = os.path.join(fragment_path, 'features.csv')
        augmented_dict, augmented_dict_filenames = augment_fragment(
            fragment_images_base_path=fragment_images_path,
            original_fragment_pr_lm_df=original_fragment_pr_lm_df,
            cluster_faces=cluster_faces,
            cluster_type=cluster_type,
            clusterer_path=clusterer_path,
            reducer_path=reducer_path,
            fragment_apex=np.nan if np.isnan(apex) else int(apex - onset),
            num_augments=num_augments,
        )
        
        output_fragment_path = os.path.join(output_base_path, fragment_dir)
        os.makedirs(output_fragment_path, exist_ok=True)
        output_original_pr_lm_path = os.path.join(output_fragment_path, 'procrustes_lm_original.csv')
        shutil.copy2(original_fragment_pr_lm_path, output_original_pr_lm_path)

        coord_columns = [column for column in original_fragment_pr_lm_df.columns if re.match(r'(n_)?[xy]\d+', column)]
        for cl_num, au_lm in augmented_dict.items():
            flat_au_lm = [frame_lm.reshape(-1) for frame_lm in au_lm]
            au_df = pd.DataFrame(flat_au_lm, columns=coord_columns)
            au_face_filename = augmented_dict_filenames[cl_num]
            au_df.insert(0, "filename", original_fragment_pr_lm_df['filename'])
            au_df.to_csv(os.path.join(output_fragment_path, f'procrustes_lm_{cl_num}_{au_face_filename[:-4]}.csv'), sep=';')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r'data\me_landmarks\casme3_spotting', help="Base path to ME dataset landmarks")
    parser.add_argument("--output", default=r'data\augmented\casme3_spotting', help="Path to save augmented fragments")
    parser.add_argument("--anno_path", default=r'data\me_landmarks\casme3_spotting\labels.xlsx', help="Path to ME dataset annotation")
    parser.add_argument("--images_base_path", default=r'data\raw\casme3_spotting', help="Path to raw dataset fragments")
    parser.add_argument("--cluster_type", default='landmarks', choices=['embeddings', 'landmarks'], help="Clustering space")
    parser.add_argument("--clusterer_path", default=r'models\face_clustering\cluster_model_landmarks_kmeans.pkl', help="Path to model for clustering")
    parser.add_argument("--reducer_path", default=r'models\face_clustering\reducer_model_landmarks.pkl', help="Path to model for dim reducing (landmarks cluster_type)")
    parser.add_argument("--faces_clusters_path", default=r'data\processed_faces\cl_celeba_hq_landmarks_kmeans.h5', help="Path to clustered faces")
    parser.add_argument("--num_augments", default=4, type=int, help="Number of augments for 1 fragment")
    args = parser.parse_args()
    random.seed(42)
    augment_dataset(
        me_lm_base_path=args.input,
        output_base_path=args.output,
        me_anno_path=args.anno_path,
        me_images_base_path=args.images_base_path,
        cluster_type=args.cluster_type,
        clusterer_path=args.clusterer_path,
        reducer_path=args.reducer_path,
        faces_clusters_path=args.faces_clusters_path,
        num_augments=args.num_augments, 
    )