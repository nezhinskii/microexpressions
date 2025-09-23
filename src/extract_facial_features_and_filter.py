import pandas as pd
import numpy as np
import argparse
import re
from umap import UMAP
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import mlflow

def get_angles_filter(df, yaw_threshold = 7, pitch_threshold = 7):
    return (np.abs(df['yaw']) < yaw_threshold) & (np.abs(df['pitch']) < pitch_threshold)

def reduce(df, out_dim, random_state = 42, verbose = True):
    reducing_columns = [column for column in df.columns if re.match(r'(n_)?[xy]\d+', column)]
    other_columns = [column for column in df.columns if column not in reducing_columns]
    X = df[reducing_columns].values
    reducer = UMAP(
        n_components=out_dim,
        random_state=random_state,
        metric='euclidean',
        verbose=verbose
    )
    embeddings = reducer.fit_transform(X)
    umap_columns = [f'x{i}' for i in range(out_dim)]
    result_df = pd.DataFrame(embeddings, columns=umap_columns)
    result_df[other_columns] = df[other_columns]
    return result_df

def get_local_outlier_factor(df, angles_filter, outliers_dim = 15, random_state = 42, verbose = True):
    reduced_df = reduce(df, outliers_dim, random_state, verbose)
    lof = LocalOutlierFactor()
    coord_columns = [column for column in reduced_df.columns if re.match(r'(n_)?[xy]\d+', column)]
    lof.fit(reduced_df[angles_filter][coord_columns])
    return -lof.negative_outlier_factor_

def get_clusters(df, angles_filter, clustering_dim = 2, random_state = 42, verbose = True):
    reduced_df = reduce(df, clustering_dim, random_state, verbose)
    dbscan = DBSCAN(min_samples = 10)
    coord_columns = [column for column in reduced_df.columns if re.match(r'(n_)?[xy]\d+', column)]
    return dbscan.fit_predict(reduced_df[angles_filter][coord_columns])

def extract_features(df):
    left_eye = [36, 37, 38, 39, 40, 41]
    left_eye_outer = 36
    left_eye_inner = 39
    right_eye = [42, 43, 44, 45, 46, 47]
    right_eye_outer = 45
    right_eye_inner = 42
    mouth = [i for i in range(48, 68)]
    nose_base = 27
    nose_tip = 30
    chin = 8
    face_upper_left = 0
    face_upper_right = 16
    face_middle_left = 3
    face_middle_right = 13
    face_bottom_left = 6
    face_bottom_right = 10

    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    left_eye_center_x = df[[f'n_x{n}' for n in left_eye]].mean(axis=1)
    left_eye_center_y = df[[f'n_y{n}' for n in left_eye]].mean(axis=1)
    right_eye_center_x = df[[f'n_x{n}' for n in right_eye]].mean(axis=1)
    right_eye_center_y = df[[f'n_y{n}' for n in right_eye]].mean(axis=1)
    # eye_distance = distance(left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y)
    eye_distance = np.abs(right_eye_center_x - left_eye_center_x)

    # left_eye_width = distance(df[f'n_x{left_eye_inner}'], df[f'n_y{left_eye_inner}'], df[f'n_x{left_eye_outer}'], df[f'n_y{left_eye_outer}'])
    left_eye_width = np.abs(df[f'n_x{left_eye_inner}'] - df[f'n_x{left_eye_outer}'])
    # right_eye_width = distance(df[f'n_x{right_eye_inner}'], df[f'n_y{right_eye_inner}'], df[f'n_x{right_eye_outer}'], df[f'n_y{right_eye_outer}'])
    right_eye_width = np.abs(df[f'n_x{right_eye_outer}'] - df[f'n_x{right_eye_inner}'])
    eye_width = (left_eye_width + right_eye_width) / 2

    # nose_length = distance(df[f'n_x{nose_base}'], df[f'n_y{nose_base}'], df[f'n_x{nose_tip}'], df[f'n_y{nose_tip}'])
    nose_length = np.abs(df[f'n_y{nose_base}'] - df[f'n_y{nose_tip}'])

    mouth_center_x = df[[f'n_x{n}' for n in mouth]].mean(axis=1)
    mouth_center_y = df[[f'n_y{n}' for n in mouth]].mean(axis=1)
    # eyes_mouth_distance = distance(mouth_center_x, mouth_center_y, (left_eye_center_x + right_eye_center_x)/2, (left_eye_center_y + right_eye_center_y)/2)
    eyes_mouth_distance = np.abs(mouth_center_y - (left_eye_center_y + right_eye_center_y) / 2)

    # nose_mouth_distance = distance(mouth_center_x, mouth_center_y, df[f'n_x{nose_tip}'], df[f'n_y{nose_tip}'])
    nose_mouth_distance = np.abs(mouth_center_y - df[f'n_y{nose_tip}'])

    #mouth_chin_distance = distance(mouth_center_x, mouth_center_y, df[f'n_x{chin}'], df[f'n_y{chin}'])
    mouth_chin_distance = np.abs(mouth_center_y - df[f'n_y{chin}'])

    # face_upper_width = distance(df[f'n_x{face_upper_left}'], df[f'n_y{face_upper_left}'], df[f'n_x{face_upper_right}'], df[f'n_y{face_upper_right}'])
    face_upper_width = np.abs(df[f'n_x{face_upper_left}'] - df[f'n_x{face_upper_right}'])

    # face_middle_width = distance(df[f'n_x{face_middle_left}'], df[f'n_y{face_middle_left}'], df[f'n_x{face_middle_right}'], df[f'n_y{face_middle_right}'])
    face_middle_width = np.abs(df[f'n_x{face_middle_left}'] - df[f'n_x{face_middle_right}'])

    # face_bottom_width = distance(df[f'n_x{face_bottom_left}'], df[f'n_y{face_bottom_left}'], df[f'n_x{face_bottom_right}'], df[f'n_y{face_bottom_right}'])
    face_bottom_width = np.abs(df[f'n_x{face_bottom_left}'] - df[f'n_x{face_bottom_right}'])

    feature_names = ['eye_distance', 'eye_width', 'nose_length', 'eyes_mouth_distance', 'nose_mouth_distance', 'mouth_chin_distance', 
                     'face_upper_width', 'face_middle_width', 'face_bottom_width']
    features = np.vstack([eye_distance, eye_width, nose_length, eyes_mouth_distance, nose_mouth_distance, mouth_chin_distance,
                          face_upper_width, face_middle_width, face_bottom_width]).T
    return feature_names, features

def extract_facial_features_and_filter(input_path, output_path, yaw_threshold = 7, pitch_threshold = 7, outliers_dim = 15, 
                                       clustering_dim = 2, lof_enabled = False, lof_threshold = 1.15, random_state = 42):
    pr_lm_df = pd.read_hdf(input_path)
    angles_filter = get_angles_filter(pr_lm_df, yaw_threshold, pitch_threshold)

    if lof_enabled:
        lof_scores = get_local_outlier_factor(pr_lm_df, angles_filter, outliers_dim, random_state)
    clusters = get_clusters(pr_lm_df, angles_filter, clustering_dim, random_state)

    values, counts = np.unique(clusters[clusters >= 0], return_counts=True)
    main_cluster = values[counts.argmax()]

    if lof_enabled:
        filtered_pr_lm_df = pr_lm_df[angles_filter][(clusters == main_cluster) & (lof_scores < lof_threshold)]
    else:
        filtered_pr_lm_df = pr_lm_df[angles_filter][clusters == main_cluster]

    feature_names, features = extract_features(filtered_pr_lm_df)
    mlflow.log_param('feature_names', feature_names)
    result_df = pd.DataFrame(features, columns = feature_names, index = filtered_pr_lm_df.index)
    result_df['filename'] = filtered_pr_lm_df['filename']
    result_df.to_hdf(output_path, key='landmarks', mode='w', format='table')
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input normalized landmarks")
    parser.add_argument("--output", required=True, help="Path to facial features df")
    parser.add_argument("--outliers_dim", type=int, default=15, help="Dimension of the reduced space for outlier search")
    parser.add_argument("--clustering_dim", type=int, default=2, help="Dimension of the reduced space for clustering")
    parser.add_argument("--yaw_threshold", type=int, default=7, help="Yaw filter threshold")
    parser.add_argument("--pitch_threshold", type=int, default=7, help="Pitch filter threshold")
    parser.add_argument("--lof_enabled", action="store_true", help="Enable LOF filtering")
    parser.add_argument("--lof_threshold", type=np.float64, default=1.1, help="LOF threshold")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    args = parser.parse_args()
        
    mlflow.set_experiment("filtering & feature extraction")
    with mlflow.start_run():
        mlflow.log_param("input", args.input)
        mlflow.log_param("output", args.output)
        mlflow.log_param("outliers_dim", args.outliers_dim)
        mlflow.log_param("clustering_dim", args.clustering_dim)
        mlflow.log_param("yaw_threshold", args.yaw_threshold)
        mlflow.log_param("pitch_threshold", args.pitch_threshold)
        mlflow.log_param("lof_enabled", args.lof_enabled)
        mlflow.log_param("lof_threshold", args.lof_threshold)
        extract_facial_features_and_filter(args.input, args.output, args.yaw_threshold, args.pitch_threshold, args.outliers_dim, 
                                        args.clustering_dim, args.lof_enabled, args.lof_threshold, args.random_state)