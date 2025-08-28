import argparse
import mlflow
from umap.parametric_umap import ParametricUMAP
from umap import UMAP
import pandas as pd
import argparse
import re
import os
import pickle
from sklearn.manifold import trustworthiness

def reduce_landmarks(input_path, output_path, type, out_dim, random_state):
    df = pd.read_hdf(input_path)
    coord_columns = [column for column in df.columns if re.match(r'[xy]\d+', column)]
    X = df[coord_columns].values
    
    if type == "umap":
        reducer = UMAP(
            n_components=out_dim,
            random_state=random_state,
            metric='euclidean',
            verbose=True
        )
    else:
        reducer = ParametricUMAP(
            n_components=out_dim,
            random_state=random_state,
            verbose=True
        )
        
    embeddings = reducer.fit_transform(X)
    
    umap_columns = [f'x{i}' for i in range(out_dim)]
    result_df = pd.DataFrame(embeddings, columns=umap_columns)
    result_df['filename'] = df['filename']
    result_df.to_hdf(output_path, key='data', mode='w')
    
    trust_score = trustworthiness(X, embeddings, n_neighbors=5)
    mlflow.log_metric("trustworthiness", trust_score)
    
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    model_filename = f"{input_basename}_{type}_{out_dim}"
    os.makedirs('models/reducers', exist_ok=True)
    if type == "umap":
        with open(f'models/reducers/{model_filename}.pkl', 'wb') as f:
            pickle.dump(reducer, f)
    else:
        reducer.save(f'models/reducers/{model_filename}')
    return result_df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input landmarks")
    parser.add_argument("--output", required=True, help="Path to reduced landmarks")
    parser.add_argument("--type", required=True, choices=["umap", "parametric_umap"], help="Reducer type")
    parser.add_argument("--out_dim", default=True, type=int, help="Output dimension")
    parser.add_argument("--random_state", default=42, type=int, help="Random state for UMAP")
    args = parser.parse_args()
    output_size = tuple(map(int, args.output_size.split('x')))

    mlflow.set_experiment("dim_reduction")
    with mlflow.start_run():
        mlflow.log_param("input_path", args.input)
        mlflow.log_param("output_path", args.output)
        mlflow.log_param("type", args.type)
        mlflow.log_param("out_dim", args.out_dim)
        mlflow.log_param("random_state", args.random_state)

        reduce_landmarks(args.input, args.output, args.type, args.out_dim, args.random_state)