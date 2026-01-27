import argparse
import os
import re
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib

def reduce_lm(df, out_dim, random_state = 42):
    reducing_columns = [column for column in df.columns if re.match(r'(n_)?[xy]\d+', column)]
    X = df[reducing_columns].values
    reducer = UMAP(
        n_components=out_dim,
        random_state=random_state,
        metric='euclidean',
    )
    reduced_features_lm = reducer.fit_transform(X)
    return reduced_features_lm

def get_filter_masks(
    df,
    max_yaw=12.0,
    max_pitch=17.0,
    min_neutral=70.0,
    lof_dim=15,
    random_state=42
):
    angles_emotions_mask =  (np.abs(df['yaw']) <= max_yaw) & \
                            (np.abs(df['pitch']) <= max_pitch) & \
                            (df['neutral_conf'] >= min_neutral)
    reduced_features_lm = reduce_lm(df, lof_dim, random_state)
    reduced_features_lm = StandardScaler().fit_transform(reduced_features_lm)
    filtered_reduced_features_lm = reduced_features_lm[angles_emotions_mask]
    lof = LocalOutlierFactor()
    lof_pred = lof.fit_predict(filtered_reduced_features_lm)
    return angles_emotions_mask, (lof_pred == 1)

def get_clusters(
    filtered_df,
    cluster_type='landmarks',
    cluster_method='kmeans',
    n_clusters=25,
    eps=0.5,
    min_samples=5,
    lm_cl_dim=20,
    random_state=42
):
    reducer = None
    if cluster_type == 'embeddings':
        emb_cols = [f'emb{i}' for i in range(512)]
        features = filtered_df[emb_cols].values.astype(np.float32)
        features = normalize(features, norm='l2')
        db_mentric = 'cosine'
    elif cluster_type == 'landmarks':
        lm_cols = [column for column in filtered_df.columns if re.match(r'(n_)?[xy]\d+', column)]
        landmarks = filtered_df[lm_cols].values.astype(np.float32)
        reducer = PCA(n_components=lm_cl_dim, random_state=random_state)
        features = reducer.fit_transform(landmarks)
        db_mentric = 'euclidean'
    else:
        raise ValueError("cluster_type must be 'embeddings' or 'landmarks'")

    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        labels = clusterer.fit_predict(features)
    elif cluster_method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=db_mentric)
        labels = clusterer.fit_predict(features)
    else:
        raise ValueError("cluster_method must be 'kmeans' or 'dbscan'")
    
    return labels, clusterer, reducer

def filter_and_cluster_faces(
    input_path=r'data/processed_faces/procrustes_processed_celeba_hq.h5',
    output_dir=r'data/processed_faces',
    input_name=r'celeba_hq',
    models_path=r'models/face_clustering',
    max_yaw=12.0,
    max_pitch=17.0,
    min_neutral=70.0,
    lof_dim=15,
    cluster_type='landmarks',
    cluster_method='kmeans',
    n_clusters=25,
    eps=0.5,
    min_samples=5,
    lm_cl_dim=20,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    
    df = pd.read_hdf(input_path, key='landmarks')

    angles_emotions_mask, outlier_mask = get_filter_masks(
        df=df,
        max_yaw=max_yaw,
        max_pitch=max_pitch,
        min_neutral=min_neutral,
        lof_dim=lof_dim,
        random_state=random_state
    )
    filtered_df = df[angles_emotions_mask][outlier_mask].copy()

    if len(filtered_df) == 0:
        raise ValueError("No faces left after filtering. Adjust thresholds.")
    
    labels, clusterer, reducer = get_clusters(
        filtered_df=filtered_df,
        cluster_type=cluster_type,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        lm_cl_dim=lm_cl_dim,
        random_state=random_state
    )
    
    if reducer:
        joblib.dump(reducer, os.path.join(models_path, f"reducer_model_{cluster_type}.pkl"))

    filtered_df['cluster_id'] = labels

    filename = f'cl_{input_name}_{cluster_type}_{cluster_method}.h5'
    output_path = os.path.join(output_dir, filename)
    filtered_df.to_hdf(output_path, key='landmarks', mode='w', format='table')
    model_path = os.path.join(models_path, f"cluster_model_{cluster_type}_{cluster_method}.pkl")
    joblib.dump(clusterer, model_path)

    print(f"Processed {len(filtered_df)} faces")
    print(f"Saved data to: {output_path}")
    print(f"Saved cluster model to: {model_path}")
    if cluster_type == 'landmarks':
        print(f"Saved reducer model to: {model_path}")

    return filtered_df

def predict_cluster_for_new_face(
    new_landmarks_or_embedding,
    clusterer,
    cluster_type='landmarks',
    reducer=None,
):    
    if isinstance(new_landmarks_or_embedding, pd.DataFrame) or isinstance(new_landmarks_or_embedding, pd.Series):
        if cluster_type == 'embeddings':
            emb_cols = [f'emb{i}' for i in range(512)]
            new_landmarks_or_embedding = new_landmarks_or_embedding[emb_cols].values.astype(np.float32)
        elif cluster_type == 'landmarks':
            all_cols = new_landmarks_or_embedding.columns if isinstance(new_landmarks_or_embedding, pd.DataFrame) else new_landmarks_or_embedding.index
            lm_cols = [column for column in all_cols if re.match(r'(n_)?[xy]\d+', column)]
            new_landmarks_or_embedding = new_landmarks_or_embedding[lm_cols].values.astype(np.float32)
    
    if len(new_landmarks_or_embedding.shape) < 2:
        new_landmarks_or_embedding = new_landmarks_or_embedding.reshape(1, -1)
    
    if cluster_type == 'landmarks':
        if reducer is None:
            raise ValueError("reducer required for landmarks")
        features = reducer.transform(new_landmarks_or_embedding)
    else:
        features = normalize(new_landmarks_or_embedding, norm='l2')
        
    return clusterer.predict(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter faces and cluster by embeddings or normalized landmarks")
    parser.add_argument("--input", default=r"data/processed_faces/procrustes_processed_celeba_hq.h5", help="Input .h5 after procrustes")
    parser.add_argument("--output_dir", default=r"data/processed_faces", help="Output directory (filename generated automatically)")
    parser.add_argument("--input_name", default="celeba_hq", help="Input dataset name")
    parser.add_argument("--models_path", default=r"models/face_clustering", help="Saved models path")
    parser.add_argument("--max_yaw", type=float, default=12.0, help="Max |yaw| degrees")
    parser.add_argument("--max_pitch", type=float, default=17.0, help="Max |pitch| degrees")
    parser.add_argument("--min_neutral", type=float, default=70.0, help="Min neutral confidence (0-100)")
    parser.add_argument("--lof_dim", type=float, default=15, help="Dimension of the reduced space for outlier search")
    parser.add_argument("--cluster_type", default="landmarks", choices=['embeddings', 'landmarks'], help="Clustering space")
    parser.add_argument("--cluster_method", default="kmeans", choices=['kmeans', 'dbscan'], help="Method")
    parser.add_argument("--n_clusters", type=int, default=25, help="For KMeans")
    parser.add_argument("--eps", type=float, default=0.5, help="For DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="For DBSCAN")
    parser.add_argument("--lm_cl_dim", type=int, default=20, help="Dimension of the reduced space for clustering by landmarks")
    parser.add_argument("--random_state", type=int, default=42, help="Seed")

    args = parser.parse_args()

    filter_and_cluster_faces(
        input_path=args.input,
        output_dir=args.output_dir,
        input_name=args.input_name,
        models_path=args.models_path,
        max_yaw=args.max_yaw,
        max_pitch=args.max_pitch,
        min_neutral=args.min_neutral,
        lof_dim=args.lof_dim,
        cluster_type=args.cluster_type,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        lm_cl_dim=args.lm_cl_dim,
        random_state=args.random_state
    )