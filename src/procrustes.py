import pandas as pd
import numpy as np
import re
import argparse
from scipy.linalg import orthogonal_procrustes

def load_meanface(meanface_path=r"models\meanface.txt"):
    with open(meanface_path) as f:
        line = f.readlines()[0]
    vals = [float(x) for x in line.strip().split()]
    points = np.array(vals).reshape(-1, 2)
    points -= np.mean(points, axis=0)
    return points

def compute_similarity_params(target_points, reference_points, slice_for_alignment=None, allow_reflection=True):
    target_centroid = np.mean(target_points, axis=0)
    ref_centroid = np.mean(reference_points, axis=0)

    centered_target = target_points - target_centroid
    centered_ref = reference_points - ref_centroid

    if slice_for_alignment is not None:
        centered_target = centered_target[slice_for_alignment]
        centered_ref = centered_ref[slice_for_alignment]

    R_raw, _ = orthogonal_procrustes(centered_target, centered_ref)
    R = R_raw.T

    if not allow_reflection and np.linalg.det(R) < 0:
        R[:, -1] *= -1

    norm_target = np.linalg.norm(centered_target)
    norm_ref = np.linalg.norm(centered_ref)
    scale = norm_ref / norm_target if norm_target > 1e-8 else 1.0
    t = ref_centroid - scale * (target_centroid @ R)

    return scale, R, t


def apply_similarity_transform(points, scale, R, t):
    return scale * (points @ R) + t

def normalize_single_face_with_params(target_points, reference_points, slice_for_alignment=None, allow_reflection=True):
    scale, R, t = compute_similarity_params(
        target_points, reference_points, slice_for_alignment, allow_reflection
    )
    normalized_points = apply_similarity_transform(target_points, scale, R, t)
    return normalized_points, scale, R, t

def normalize_landmarks(input_path, output_path, meanface_path, slice_for_alignment=None, allow_reflection=True, center_meanface_to_zero=True):
    reference_points = load_meanface(meanface_path, center_to_zero=center_meanface_to_zero)
    
    df = pd.read_hdf(input_path)
    coord_columns = [col for col in df.columns if re.match(r'[xy]\d+', col)]
    other_columns = [col for col in df.columns if col not in coord_columns + ['filename']]
    norm_coord_columns = ['n_' + col for col in coord_columns]
    
    normalized_rows = []
    for _, row in df.iterrows():
        target_points = row[coord_columns].values.astype(np.float32).reshape(68, 2)
        
        normalized_points, _, _, _ = normalize_single_face_with_params(
            target_points, reference_points, allow_reflection=allow_reflection,
            slice_for_alignment=slice_for_alignment
        )
        
        flat_normalized = normalized_points.reshape(-1)
        new_row = [row['filename']] + flat_normalized.tolist() + row[other_columns].tolist()
        normalized_rows.append(new_row)
    
    result_df = pd.DataFrame(normalized_rows, columns=['filename'] + norm_coord_columns + other_columns)
    result_df.to_hdf(output_path, key='landmarks', mode='w', format='table')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"data\landmarks\lm_celeba_hq_starnet.h5", help="Path to input landmarks")
    parser.add_argument("--output", default=r"data\landmarks\pr_lm_celeba_hq_starnet.h5", help="Path to normalized landmarks")
    parser.add_argument("--meanface_path", default=r"models\meanface.txt", help="Path to meanface.txt")
    args = parser.parse_args()
        
    normalize_landmarks(args.input, args.output, args.meanface_path)