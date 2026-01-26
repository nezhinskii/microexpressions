import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from deepface import DeepFace
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

# Reuse from landmarks.py
from landmarks import (
    ImageDataset,
    get_transforms,
    load_model,
    run_landmark_model,
    get_angles,
)

def apply_landmark_transform(batch_images, model_transforms):
    if model_transforms is None:  # starnet
        return batch_images
    # For facexformer/pipnet: convert each to PIL RGB, apply transform, stack tensors
    batch_tensors = []
    for img in batch_images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = model_transforms(img_pil)
        batch_tensors.append(tensor)
    return torch.stack(batch_tensors)

def process_landmarks(batch_images, model, extra_data, model_type, model_transforms, device):
    if len(batch_images.shape) < 4:
        batch_images = batch_images.unsqueeze(0)
    batch_transformed = apply_landmark_transform(batch_images, model_transforms)
    batch_transformed = batch_transformed.to(device)
    landmark_output = run_landmark_model(model_type, model, batch_transformed, extra_data, device)
    return landmark_output

def process_emotions(batch_images):
    if len(batch_images.shape) < 4:
        batch_images = batch_images.unsqueeze(0)
    results = DeepFace.analyze(
        batch_images,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="yunet",
        silent=True
    )
    return [res[0]['emotion']['neutral'] if res[0]['face_confidence'] > 0.5 else None for res in results]

def process_embeddings(batch_images):
    if len(batch_images.shape) < 4:
        batch_images = batch_images.unsqueeze(0)
    results = DeepFace.represent(
        batch_images, 
        enforce_detection=False, 
        model_name='Facenet512', 
        detector_backend='yunet'
    )
    return [res[0]['embedding'] if res[0]['face_confidence'] > 0.5 else None for res in results]

def process_angles(batch_landmarks, batch_images):
    results = []
    for landmarks, image in zip(batch_landmarks, batch_images):
        if landmarks is None:
            results.append(None)
            continue
        if image.shape[0] == 3:
            img_size = image.shape[1:]
        else:
            img_size = image.shape[:2]
        results.append(get_angles(landmarks, img_size))
    return results

def preprocess_faces(
    input_dir=r"data/raw/celeba_hq",
    input_name="celeba_hq",
    output_dir=r"data/processed_faces",
    model_path=r"models/300W_STARLoss_NME_2_87.pkl",
    model_type=r"starnet",
    meanface_path=r"models/meanface.txt",
    face_detector_path=r"models/shape_predictor_68_face_landmarks.dat",
    batch_size=32,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = ImageDataset(input_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, extra_data = load_model(
        model_type=model_type,
        model_path=model_path,
        device=device,
        face_detector_path=face_detector_path,
        meanface_path=meanface_path,
    )
    model_transforms = get_transforms(
        model_type=model_type
    )

    df_rows = []
    for batch_images, batch_filenames in tqdm(dataloader, desc="Processing batches"):
        batch_images_np = batch_images.numpy()
        batch_landmarks = process_landmarks(
            batch_images=batch_images,
            model=model,
            extra_data=extra_data,
            model_type=model_type,
            model_transforms=model_transforms,
            device=device
        )
        batch_angles = process_angles(batch_landmarks, batch_images)
        batch_emotions = process_emotions(batch_images_np)
        batch_embeddings = process_embeddings(batch_images_np)
        
        for i in range(len(batch_images)):
            if (batch_landmarks[i] is None) or (batch_angles[i] is None) or (batch_emotions[i] is None) or (batch_embeddings[i] is None):
                continue
            df_row = [
                batch_filenames[i],
                *batch_landmarks[i].flatten().tolist(),
                *batch_angles[i],
                batch_emotions[i],
                *batch_embeddings[i]
            ]
            df_rows.append(df_row)

    df_columns = ['filename']
    for i in range(68):
        df_columns.append(f'x{i}')
        df_columns.append(f'y{i}')
    df_columns.extend(['yaw', 'pitch', 'roll', 'neutral_conf'])
    df_columns.extend([f'emb{i}' for i in range(512)])
    df = pd.DataFrame(df_rows, columns=df_columns)

    output_file = os.path.join(output_dir, f"processed_{input_name}.h5")
    df.to_hdf(output_file, key='landmarks', mode='w', format='table')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks, headpose, emotion and Facenet512 embeddings")
    parser.add_argument("--input", default=r"data/raw/celeba_hq", help="Path to input images directory")
    parser.add_argument("--input_name", default="celeba_hq", help="Name of the dataset (used in output filename)")
    parser.add_argument("--output", default=r"data/processed_faces", help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (cuda:0, cpu, etc.)")

    args = parser.parse_args()

    preprocess_faces(
        input_dir=args.input,
        input_name=args.input_name,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )