import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image
import base64
from io import BytesIO
import os
import numpy as np
import cv2

def plot_landmarks_cv2(image_path, landmarks, radius=1):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    for i in range(len(landmarks) // 2): 
        x = int(((landmarks[f"x{i}"] + 1) / 2) * img_width)
        y = int(((landmarks[f"y{i}"] + 1) / 2)  * img_height)
        cv2.circle(img, (x, y), radius=radius, color=(0, 255, 0), thickness=-1)
    return img

def plot_emb_with_images(df: pd.DataFrame, lm_df: pd.DataFrame, n_clusters=20, n_reps_per_cluster=1, random_state=42, image_dir=r"data\processed\aligned_celeba_hq"):
    df = df.copy()
    has_clusters = 'cluster' in df.columns
    if not has_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        df['cluster'] = kmeans.fit_predict(df[['x0', 'x1']])
        
    representatives = []
    for cluster in range(n_clusters):
        cluster_points = df[df['cluster'] == cluster]
        if not cluster_points.empty:
            try:
                rep_points = cluster_points.sample(n=n_reps_per_cluster, random_state=random_state)
                representatives.extend(rep_points.to_dict('records'))
            except ValueError:
                representatives.extend(cluster_points.to_dict('records'))
    reps_df = pd.DataFrame(representatives)
    
    
    fig = px.scatter(
        df,
        x='x0',
        y='x1',
        color='cluster' if has_clusters else None,
        hover_name='filename',
        opacity=0.4,
        width=1200,
        height=900,
        title='2D Embedding Space with Representative Images'
    )
    
    for _, rep in reps_df.iterrows():
        x, y, filename = rep['x0'], rep['x1'], rep['filename']
        lm_row = lm_df[lm_df['filename'] == filename].iloc[0]
        image_path = os.path.join(image_dir, filename)
        img = plot_landmarks_cv2(image_path, lm_row, radius=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        def crop_black_borders_and_resize(img_np, target_size=100):
            non_black = np.any(img_np > 15, axis=2)
            rows, cols = np.where(non_black)
            top, bottom = rows.min(), rows.max() + 1
            left, right = cols.min(), cols.max() + 1
            img_cropped = img_np[top:bottom, left:right]
            height, width = img_cropped.shape[:2]
            aspect_ratio = width / height
            if width > height:
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(target_size * aspect_ratio)
            img_pil = Image.fromarray(img_cropped)
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return img_resized
        
        img_pil = crop_black_borders_and_resize(img, target_size=100)
        
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_str}",
                xref="x", yref="y",
                x=x, y=y,
                sizex=1.2, sizey=1.2,
                xanchor="center", yanchor="middle"
            )
        )
    fig.show()
