import argparse
import os
import re
import argparse
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.preprocessing import normalize

def reduce(input_path, output_path, random_state):
    df = pd.read_hdf(input_path)
    
    def get_reduced_features(X, metric):
        reducer = UMAP(
            n_components=2,
            random_state=random_state,
            metric=metric,
            verbose=True
        )
        reduced_values = reducer.fit_transform(X)
        return reduced_values
    
    emb_cols = [f'emb{i}' for i in range(512)]
    features_emb = df[emb_cols].values.astype(np.float32)
    features_emb = normalize(features_emb, norm='l2')
    reduced_emb = get_reduced_features(features_emb, 'cosine')
    reduced_emb_df = pd.DataFrame(reduced_emb, columns=['x0', 'x1'])
    reduced_emb_df['filename'] = df['filename']
    reduced_emb_df.to_hdf(os.path.join(output_path, f'umap2_emb_{os.path.basename(input_path)}'), key='data', mode='w')
    
    lm_cols = [column for column in df.columns if re.match(r'(n_)?[xy]\d+', column)]
    features_lm = df[lm_cols].values.astype(np.float32)
    reduced_lm = get_reduced_features(features_lm, 'euclidean')
    reduced_lm_df = pd.DataFrame(reduced_lm, columns=['x0', 'x1'])
    reduced_lm_df['filename'] = df['filename']
    reduced_lm_df.to_hdf(os.path.join(output_path, f'umap2_lm_{os.path.basename(input_path)}'), key='data', mode='w')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"data\processed_faces\procrustes_processed_celeba_hq.h5", help="Path to input df")
    parser.add_argument("--output", default=r"data\processed_faces", help="Path to reduced features")
    parser.add_argument("--random_state", default=42, type=int, help="Random state for UMAP")
    args = parser.parse_args()

    reduce(args.input, args.output, args.random_state)