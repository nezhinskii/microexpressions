import os
import pandas as pd
import cv2
import numpy as np
import torch
import re
import onnxruntime as ort
import argparse
from tqdm import tqdm
from PIL import Image
from alignment import align_face, detect_faces, create_session
from landmarks import run_landmark_model, load_model, get_transforms
from procrustes import normalize_points, procrustes_normalization
from extract_facial_features_and_filter import extract_features

def align_fragment_frames(fragment_path, detect_model, output_size, aligned_path, input_name=None, save_aligned=False):
    aligned_frames = []
    if isinstance(aligned_path, str) and save_aligned:
        os.makedirs(aligned_path, exist_ok=True)
    for frame_name in os.listdir(fragment_path):
        frame_path = os.path.join(fragment_path, frame_name)
        image, detections = detect_faces(frame_path, detect_model, input_name)
        aligned_image = align_face(image, np.array(detections[0]["landmarks"]), detections[0]["bbox"], output_size)
        if isinstance(aligned_path, str) and save_aligned:
            aligned_frame_path = os.path.join(aligned_path, frame_name)
            cv2.imwrite(aligned_frame_path, cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        aligned_frames.append(aligned_image)
    return aligned_frames

def proctrustes_fragment_frames(meanface_path, fragment_landmarks):
    with open(meanface_path) as f:
        meanface = f.readlines()[0]
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    normalized_meanface, _, _ = normalize_points(meanface)

    onset_lm = fragment_landmarks[0]
    target_points_slice = slice(17, 68)
    _, fixed_centroid, fixed_scale, fixed_R = procrustes_normalization(onset_lm.numpy(), normalized_meanface, target_points_slice)

    def apply_fixed_procrustes(frame_points, fixed_centroid, fixed_scale, fixed_R):
        centered = frame_points - fixed_centroid
        scaled = centered / fixed_scale
        normalized = scaled.dot(fixed_R)
        return normalized
    
    procrustes_fragment_lm = [
        apply_fixed_procrustes(frame_lm.numpy(), fixed_centroid, fixed_scale, fixed_R) 
        for frame_lm in fragment_landmarks
        ]
    return procrustes_fragment_lm

def process_fragment(
    base_path:str, 
    fragment_data:pd.Series, 
    detect_session:ort.InferenceSession, 
    lm_model_type:str, 
    lm_model, 
    lm_model_transforms, 
    lm_extra_data, 
    meanface_path: str,
    aligned_size:tuple[int, int] = (256, 256), 
    device:str = 'cuda:0', 
):
    subject=fragment_data['Subject']; filename=fragment_data['Filename']; onset=fragment_data['Onset']
    fragment_dir = subject + '_' + filename + '_' + str(onset)
    fragment_path = os.path.join(base_path, fragment_dir)
    fragment_filenames = os.listdir(fragment_path)

    # alignment
    aligned_dir = fragment_dir + '_aligned'
    aligned_path = os.path.join(base_path, aligned_dir)
    input_name = detect_session.get_inputs()[0].name
    aligned_frames = align_fragment_frames(fragment_path, detect_session, aligned_size, aligned_path, input_name, True)

    # landmark
    if lm_model_transforms is None:
        transformed_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in aligned_frames]
    else:
        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in aligned_frames]
        transformed_frames = torch.stack([lm_model_transforms(frame) for frame in pil_frames])
    fragment_landmarks = run_landmark_model(lm_model_type, lm_model, transformed_frames, lm_extra_data, device)
    raw_lm_data = []
    for filename, landmarks in zip(fragment_filenames, fragment_landmarks):
        if landmarks is None:
            raise ValueError(f"No face found on {filename}")
        landmarks = landmarks.cpu().numpy().flatten()
        result = {'filename': filename}
        for i in range(68):
            result[f'x{i}'] = landmarks[2 * i]
            result[f'y{i}'] = landmarks[2 * i + 1]
        raw_lm_data.append(result)
    raw_lm_df = pd.DataFrame(raw_lm_data)

    # procrustes
    procrustes_fragment_lm = proctrustes_fragment_frames(meanface_path, fragment_landmarks)
    coord_columns = [column for column in raw_lm_df.columns if re.match(r'[xy]\d+', column)]
    norm_coord_columns = ['n_' + column for column in coord_columns]
    pr_lm_data = []
    for filename, pr_landmarks in zip(fragment_filenames, procrustes_fragment_lm):
        normalized_flat = pr_landmarks.reshape(-1)
        pr_lm_data.append([filename] + normalized_flat.tolist())
    pr_lm_df = pd.DataFrame(pr_lm_data, columns=['filename'] + norm_coord_columns)

    # extract features
    feature_names, features = extract_features(pr_lm_df)
    features_df = pd.DataFrame(features, columns = feature_names, index = pr_lm_df.index)
    features_df['filename'] = pr_lm_df['filename']

    return fragment_dir, raw_lm_df, pr_lm_df, features_df

def find_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            return os.path.join(folder_path, file)

def process_dataset(
    base_path:str=r'data\raw\casme3_partA',
    output_base_path:str=r'data\me_landmarks\casme3',
    detect_model_path:str=r'models\yolov6s_face.onnx',
    lm_model_type:str=r'pipnet',
    lm_model_path:str=r'models\pipnet.pth',
    face_detector_path:str=r'models\shape_predictor_68_face_landmarks.dat',
    meanface_path:str=r'models\meanface.txt',
    aligned_size:tuple[int, int]=(256, 256), 
    device:str='cuda:0',
    fps:int=30
):
    detect_session = create_session(detect_model_path)
    lm_model, lm_extra_data = load_model(lm_model_type, lm_model_path, device, face_detector_path, meanface_path)
    lm_model_transforms = get_transforms(lm_model_type)
    labels_path = find_with_extension(base_path, 'xlsx')
    casme_df = pd.read_excel(labels_path)
    
    fragments_path = os.path.join(base_path, 'frame')
    for index, row in tqdm(casme_df.iterrows(), total=len(casme_df)):
        subject=row['Subject']; filename=row['Filename']; onset=row['Onset']; offset=row['Offset']
        if (offset - onset > 2 * fps):
            continue
        fragment_dir = subject + '_' + filename + '_' + str(onset)
        try:
            fragment_dir, raw_lm_df, pr_lm_df, features_df = process_fragment(
                base_path=fragments_path, 
                fragment_data=row, 
                detect_session=detect_session, 
                lm_model_type=lm_model_type, 
                lm_model=lm_model, 
                lm_model_transforms=lm_model_transforms, 
                lm_extra_data=lm_extra_data, 
                meanface_path=meanface_path,
                aligned_size=aligned_size, 
                device=device, 
            )
            output_fragment_path = os.path.join(output_base_path, fragment_dir)
            os.makedirs(output_fragment_path, exist_ok=True)
            raw_lm_df.to_csv(os.path.join(output_fragment_path, "raw_landmarks.csv"), sep=';')
            pr_lm_df.to_csv(os.path.join(output_fragment_path, "procrustes_lm.csv"), sep=';')
            features_df.to_csv(os.path.join(output_fragment_path, "features.csv"), sep=';')
        except Exception as e:
            print(f"ERROR on fragment {fragment_dir}: {e}.\nSkip...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r'data\raw\casme3_partA',
                        help="Base path to ME dataset")
    parser.add_argument("--output", default=r'data\me_landmarks\casme3',
                        help="Path to save processed fragments")
    parser.add_argument("--detection_model", default=r'models\yolov6s_face.onnx',
                        help="Path to detection model (YOLO face)")
    parser.add_argument("--lm_model_type", default='pipnet',
                        choices=["facexformer", "pipnet", "starnet"],
                        help="Landmark model type")
    parser.add_argument("--lm_model", default=r'models\pipnet.pth',
                        help="Path to landmark model")
    parser.add_argument("--face_detector_path", default=r'models\shape_predictor_68_face_landmarks.dat', 
                        help="Path to face detector (for starnet)")
    parser.add_argument("--meanface_path", default=r'models\meanface.txt',
                        help="Path to meanface.txt (for pipnet)")
    parser.add_argument("--aligned_size", default="256x256",
                        help="Output size as WIDTHxHEIGHT (e.g. 256x256)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to run models on (e.g. cuda:0 or cpu)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for processing video fragments")
    args = parser.parse_args()
    
    if isinstance(args.aligned_size, str):
        aligned_size = tuple(map(int, args.aligned_size.split('x')))
    else:
        aligned_size = args.aligned_size
    
    process_dataset(
        base_path=args.input,
        output_base_path=args.output,
        detect_model_path=args.detection_model,
        lm_model_type=args.lm_model_type,
        lm_model_path=args.lm_model,
        face_detector_path=args.face_detector_path,
        meanface_path=args.meanface_path,
        aligned_size=aligned_size,
        device=args.device,
        fps=args.fps
    )