import os
import shutil
import pandas as pd
import numpy as np
import re
import argparse
import random
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from types_of_faces import get_fragment_features_center

def find_farthest_clusters(num_farthest, fragment_features_center, centers_path=r'models\types_of_faces\centers_df.csv', k_nearest=9):
    centers_df = pd.read_csv(centers_path)
    id_center_dict = {k: np.array(list(v.values())) for k, v in centers_df.set_index('face_type').to_dict('index').items()}

    distances_to_center = {}
    for id, center in id_center_dict.items():
        dist = np.linalg.norm(center - fragment_features_center)
        distances_to_center[id] = dist

    sorted_clusters_by_distance = sorted(distances_to_center.items(), key=lambda x: x[1])
    nearest_clusters = dict(sorted_clusters_by_distance[:k_nearest])
    remaining_centers = {id: center for id, center in id_center_dict.items() if id not in nearest_clusters}
    remaining_ids = list(remaining_centers.keys())
    selected_indices = random.sample(remaining_ids, num_farthest)
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
    mean_faces:dict, 
    types_of_faces_scaler_path:str,
    types_of_faces_centers_path:str,
    num_augments:int=4, 
    smoothing:float=0.01
):
    fragment_features_center = get_fragment_features_center(original_fragment_features_df, types_of_faces_scaler_path)
    farthest_clusters = find_farthest_clusters(num_augments, fragment_features_center, types_of_faces_centers_path)

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
        target_points_slice = slice(17, 68)
        interpolator = fit_tps(onset_points[target_points_slice], target_points[target_points_slice], smoothing)
        augmented_sequence = []
        for frame in procrustes_fragment_lm:
            augmented_frame = apply_tps(interpolator, frame)
            augmented_sequence.append(augmented_frame)
        augmented_dict[cl_id] = augmented_sequence
    return augmented_dict, procrustes_fragment_lm

def augment_dataset(
    me_lm_base_path:str=r'data\me_landmarks\casme3',
    output_base_path:str=r'data\augmented\casme3',
    me_anno_path:str=r'data\me_landmarks\casme3\labels.xlsx',
    typed_faces_path:str=r'data\landmarks\typed_features_celeba_hq_pipnet.h5',
    procrustes_faces_lm_path:str=r'data\landmarks\pr_lm_celeba_hq_pipnet.h5',
    types_of_faces_model_base_path:str=r'models\types_of_faces',
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
        
    typed_faces_df = pd.read_hdf(typed_faces_path)
    procrustes_lm_df = pd.read_hdf(procrustes_faces_lm_path)
    mean_faces = prepare_mean_faces(typed_faces_df, procrustes_lm_df)

    types_of_faces_scaler_path = os.path.join(types_of_faces_model_base_path, 'scaler.pkl')
    types_of_faces_centers_path = os.path.join(types_of_faces_model_base_path, 'centers_df.csv')

    fragment_dirs = set(os.listdir(me_lm_base_path))

    for index, row in tqdm(me_df.iterrows(), total=len(me_df)):
        subject=row['Subject']; filename=row['Filename']; onset=row['Onset']; offset=row['Offset']
        fragment_dir = subject + '_' + filename + '_' + str(onset)
        if fragment_dir not in fragment_dirs:
            continue
        fragment_path = os.path.join(me_lm_base_path, fragment_dir)

        original_fragment_pr_lm_path = os.path.join(fragment_path, 'procrustes_lm.csv')
        original_fragment_pr_lm_df = pd.read_csv(original_fragment_pr_lm_path, sep=';', index_col=0)
        original_fragment_features_path = os.path.join(fragment_path, 'features.csv')
        augmented_dict, procrustes_fragment_lm = augment_fragment(
            original_fragment_pr_lm_df=original_fragment_pr_lm_df,
            original_fragment_features_df=pd.read_csv(original_fragment_features_path, sep=';', index_col=0), 
            mean_faces=mean_faces, 
            types_of_faces_scaler_path=types_of_faces_scaler_path,
            types_of_faces_centers_path=types_of_faces_centers_path,
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
            au_df.insert(0, "filename", original_fragment_pr_lm_df['filename'])
            au_df.to_csv(os.path.join(output_fragment_path, f'procrustes_lm_{cl_num}.csv'), sep=';')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r'data\me_landmarks\casme3', help="Base path to ME dataset landmarks")
    parser.add_argument("--output", default=r'data\augmented\casme3', help="Path to save augmented fragments")
    parser.add_argument("--anno_path", default=r'data\me_landmarks\casme3\labels.xlsx', help="Path to ME dataset annotation")
    parser.add_argument("--typed_faces_path", default=r'data\landmarks\typed_features_celeba_hq_starnet.h5', help="Path to dataset with types of faces")
    parser.add_argument("--pr_faces_lm_path", default=r'data\landmarks\pr_lm_celeba_hq_starnet.h5', help="Path to dataset with faces landmarks")
    parser.add_argument("--types_of_faces_model", default=r'models\types_of_faces', help="Path to model for face typization")
    parser.add_argument("--num_augments", default=4, type=int, help="Number of augments for 1 fragment")
    parser.add_argument("--smoothing", default=0.01, type=float, help="smoothing value for TPS transform")
    args = parser.parse_args()
    random.seed(42)
    augment_dataset(
        me_lm_base_path=args.input,
        output_base_path=args.output,
        me_anno_path=args.anno_path,
        typed_faces_path=args.typed_faces_path,
        procrustes_faces_lm_path=args.pr_faces_lm_path,
        types_of_faces_model_base_path=args.types_of_faces_model,
        num_augments=args.num_augments, 
        smoothing=args.smoothing
    )