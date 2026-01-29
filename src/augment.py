import os
import shutil
import re
import joblib
import cv2
import random
import argparse
import numpy as np
import pandas as pd
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

def augment_fragment(
    original_fragment_pr_lm_df:pd.DataFrame,
    fragment_images_base_path:str,
    cluster_faces:dict,
    cluster_type:str,
    clusterer_path:str,
    reducer_path:str,
    num_augments:int=4, 
    smoothing:float=0.015
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
        
    regions = {
        'contour': slice(0, 17),
        'brows': slice(17, 27),
        'nose': slice(27, 36),
        'eyes': slice(36, 48),
        'mouth': slice(48, 68),
    }
    
    def calculate_ear(eye_points):
        # eye_points: 6 pts per eye (e.g., left: 36-41)
        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        vert1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vert2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horiz = np.linalg.norm(eye_points[0] - eye_points[3])
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)
    
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
    neutral_src = fragment_landmarks[neutral_idx]

    augmented_dict = {}
    augmented_dict_filenames = {}
    for cl_id in target_clusters:
        target_cluster_faces = cluster_faces[cl_id]
        target_face_row = random.choice(target_cluster_faces)
        target_face_filename = target_face_row['filename']
        target_face_points = target_face_row[coord_columns].to_numpy().astype(np.float32).reshape(-1, 2)
        
        aug_neutral = target_face_points.copy()
        augmented_sequence = []
        for i in range(len(fragment_landmarks)):
            if  i == neutral_idx:
                augmented_sequence.append(aug_neutral)
                continue
            aug_frame = aug_neutral.copy()
            curr_src = fragment_landmarks[i]
            # TPS per zone
            for reg_name, reg_slice in regions.items():
                delta_mapper = RBFInterpolator(
                    neutral_src[reg_slice],
                    curr_src[reg_slice],
                    kernel='thin_plate_spline',
                    smoothing=smoothing
                )
                aug_frame[reg_slice] = delta_mapper(aug_neutral[reg_slice])
            augmented_sequence.append(aug_frame)
            
        if len(augmented_sequence) > 3:
            aug_array = np.array(augmented_sequence)
            for pt in range(68):
                for dim in range(2):
                    aug_array[:, pt, dim] = savgol_filter(aug_array[:, pt, dim], window_length=3, polyorder=2)
            augmented_sequence = [fr for fr in aug_array]
            
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
    smoothing:float=0.015
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
    # for index, row in tqdm(me_df.iterrows(), total=len(me_df)):
    me_df = pd.read_excel(me_anno_path)
    rows=list(me_df.iterrows())
    random.seed(42)
    random.shuffle(rows)
    for index, row in rows[:15]:
        subject=row['Subject']; filename=row['Filename']; onset=row['Onset']; offset=row['Offset']
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
            num_augments=num_augments, 
            smoothing=smoothing
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
    parser.add_argument("--smoothing", default=0.01, type=float, help="smoothing value for TPS transform")
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
        smoothing=args.smoothing
    )