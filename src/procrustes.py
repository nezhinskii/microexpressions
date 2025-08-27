from scipy.linalg import orthogonal_procrustes
import pandas as pd
import numpy as np
import argparse
import re

def normalize_reference(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = np.sqrt(np.sum(centered ** 2))
    return centered / scale

def procrustes_normalization(target_points, normalized_reference):
    tar_centroid = np.mean(target_points, axis=0)
    centered_tar = target_points - tar_centroid
    tar_scale = np.sqrt(np.sum(centered_tar ** 2))
    scaled_tar = centered_tar / tar_scale
    
    R, _ = orthogonal_procrustes(scaled_tar, normalized_reference)
    normalized_target = scaled_tar.dot(R)
    return normalized_target

def normalize_landmarks(input_path, output_path, meanface_path):
    with open(meanface_path) as f:
        meanface = f.readlines()[0]
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    normalized_meanface = normalize_reference(meanface)
    
    df = pd.read_hdf(input_path)
    coord_columns = [column for column in df.columns if re.match('[xy]\d+', column)]
    norm_coord_columns = ['n_' + column for column in coord_columns]
    normalized_data = []
    for _, row in df.iterrows():
        points = row[coord_columns].values.astype(np.float32).reshape(68, 2)
        normalized_points = procrustes_normalization(points, normalized_meanface)
        normalized_flat = normalized_points.reshape(-1)
        normalized_data.append([row['filename']] + normalized_flat.tolist())
    result_df = pd.DataFrame(normalized_data, columns=['filename'] + norm_coord_columns)
    result_df.to_hdf(output_path, key='landmarks', mode='w', format='table')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input landmarks")
    parser.add_argument("--output", required=True, help="Path to normalized landmarks")
    parser.add_argument("--meanface_path", required=True, help="Path to meanface.txt")
    args = parser.parse_args()
        
    normalize_landmarks(args.input, args.output, args.meanface_path)