import pandas as pd
import numpy as np
import argparse
import re
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib

CLUST_COLUMNS = ['eye_distance', 'eyes_mouth_distance']

def get_fragment_features_center(fragment_features_df: pd.DataFrame, scaler_path:str = r'models\types_of_faces\scaler.pkl'):
    scaler = joblib.load(scaler_path)
    X = fragment_features_df[CLUST_COLUMNS]
    X_scaled = scaler.transform(X)
    return X_scaled.mean(axis=0)

def get_cells_for_new_points(new_df, scaler_path = r'models\types_of_faces\scaler.pkl', quantiles_path = r'models\types_of_faces\quantiles_df.csv'):
    scaler = joblib.load(scaler_path)
    X = new_df[CLUST_COLUMNS]
    X_scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=CLUST_COLUMNS)
    quantiles_df = pd.read_csv(quantiles_path)

    x_quantiles = quantiles_df['x_quantiles'].to_numpy()
    x_quantiles = x_quantiles[~np.isnan(x_quantiles)]
    y_quantiles = quantiles_df['y_quantiles'].to_numpy()
    y_quantiles = y_quantiles[~np.isnan(y_quantiles)]

    cells = np.full(len(new_df), -1, dtype=int)
    for i in range(len(x_quantiles) - 1):
        for j in range(len(y_quantiles) - 1):
            x_min, x_max = x_quantiles[i], x_quantiles[i + 1]
            y_min, y_max = y_quantiles[j], y_quantiles[j + 1]
            in_cell = (
                (df_scaled.iloc[:, 0] >= x_min) & (df_scaled.iloc[:, 0] < x_max) &
                (df_scaled.iloc[:, 1] >= y_min) & (df_scaled.iloc[:, 1] < y_max)
            )
            cells[in_cell] = i * (len(y_quantiles) - 1) + j
    return cells

def get_cells(df, n_sigma=3, axis_cells=5):
    X = df[CLUST_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=CLUST_COLUMNS)
    joblib.dump(scaler, r'models\types_of_faces\scaler.pkl')

    is_outlier = (np.abs(df_scaled) > n_sigma).any(axis=1)
    non_outliers = df_scaled[~is_outlier]
    x_quantiles = np.percentile(non_outliers.iloc[:, 0], np.linspace(0, 100, axis_cells + 1))
    y_quantiles = np.percentile(non_outliers.iloc[:, 1], np.linspace(0, 100, axis_cells + 1))

    cells = np.full(len(df), -1, dtype=int)
    centers = []
    for i in range(len(x_quantiles) - 1):
        for j in range(len(y_quantiles) - 1):
            x_min, x_max = x_quantiles[i], x_quantiles[i + 1]
            y_min, y_max = y_quantiles[j], y_quantiles[j + 1]
            in_cell = (
                (df_scaled.iloc[:, 0] >= x_min) & (df_scaled.iloc[:, 0] < x_max) &
                (df_scaled.iloc[:, 1] >= y_min) & (df_scaled.iloc[:, 1] < y_max)
            )
            cell_id = i * (len(y_quantiles) - 1) + j
            cells[in_cell] = cell_id
            centers.append([cell_id] + df_scaled[in_cell].mean().to_list())
            # centers.append([cell_id, (x_min + x_max) / 2, (y_min + y_max) / 2])
    quantiles_df = pd.DataFrame({
        'x_quantiles': pd.Series(x_quantiles),
        'y_quantiles': pd.Series(y_quantiles)
    })
    centers_df = pd.DataFrame(centers, columns=['face_type'] + [f'x_{i}' for i in range(len(centers[0]) - 1)])
    centers_df.to_csv(r'models\types_of_faces\centers_df.csv', index = False)
    quantiles_df.to_csv(r'models\types_of_faces\quantiles_df.csv', index = False)
    return cells, x_quantiles, y_quantiles

def get_faces_types(input_path, output_path, sigma_threshold, axis_cells):
    feat_df = pd.read_hdf(input_path)
    cells, x_quantiles, y_quantiles = get_cells(feat_df, sigma_threshold, axis_cells)
    feat_df['face_type'] = cells
    feat_df.to_hdf(output_path, key='landmarks', mode='w', format='table')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"data\landmarks\features_celeba_hq_starnet.h5", help="Path to facial features df")
    parser.add_argument("--output", default=r"data\landmarks\typed_features_celeba_hq_starnet.h5", help="Path to facial features df with types of faces")
    parser.add_argument("--sigma_threshold", type=np.float64, default=3, help="Sigma threshold for outliers")
    parser.add_argument("--axis_cells", type=int, default=5, help="Num cells by axis")
    args = parser.parse_args()
        
    mlflow.set_experiment("identifying types of faces")
    with mlflow.start_run():
        mlflow.log_param("input", args.input)
        mlflow.log_param("output", args.output)
        mlflow.log_param("sigma_threshold", args.sigma_threshold)
        mlflow.log_param("axis_cells", args.axis_cells)
        get_faces_types(args.input, args.output, args.sigma_threshold, args.axis_cells)