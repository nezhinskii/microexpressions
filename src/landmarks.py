import argparse
import os
import mlflow
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2

from facexformer import FaceXFormer
from pipnet import Pip_resnet101, get_meanface, forward_pip

class ImageDataset(Dataset):
    def __init__(self, input_path, transform=None):
        self.input_path = input_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.input_path, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, filename
    
def get_transforms(model_type):
    if model_type == "facexformer":
        return transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_type == "pipnet":
        return transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
def load_model(model_type, model_path, device, meanface_path=None):
    if model_type == "facexformer":
        model = FaceXFormer().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict_backbone'])
        model.eval()
        return model, None
    elif model_type == "pipnet":
        if meanface_path is None:
            raise ValueError("meanface_path must be provided for pipnet")
        resnet101 = models.resnet101(pretrained=True)
        model = Pip_resnet101(resnet101, num_nb=10, num_lms=68, input_size=256, net_stride=32)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(meanface_path, num_nb=10)
        reverse_index1 = torch.tensor(reverse_index1).to(device)
        reverse_index2 = torch.tensor(reverse_index2).to(device)
        extra_data = {
            "meanface_indices": meanface_indices,
            "reverse_index1": reverse_index1,
            "reverse_index2": reverse_index2,
            "max_len": max_len
        }
        model.eval()
        return model, extra_data
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
def get_landmarks_facexformer(model, batch_images, device):
    labels = {
        "segmentation": torch.zeros([len(batch_images), 224, 224]).to(device),
        "lnm_seg": torch.zeros([len(batch_images), 5, 2]).to(device),
        "landmark": torch.zeros([len(batch_images), 68, 2]).to(device),
        "headpose": torch.zeros([len(batch_images), 3]).to(device),
        "attribute": torch.zeros([len(batch_images), 40]).to(device),
        "a_g_e": torch.zeros([len(batch_images), 3]).to(device),
        'visibility': torch.zeros([len(batch_images), 29]).to(device)
    }
    task = torch.tensor([1]).repeat(len(batch_images)).to(device)
    with torch.no_grad():
        landmark_output, _, _, _, _, _, _, _ = model(batch_images, labels, task)
    return landmark_output
    
def get_landmarks_pipnet(model, batch_images, reverse_index1, reverse_index2, max_len, num_nb=10, num_lms=68, input_size=256, net_stride=32):
    with torch.no_grad():
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, _, _ = forward_pip(
            model, batch_images, None, input_size, net_stride, num_nb
        )
    
    batch_size = batch_images.size(0)
    landmark_output = []
    for idx in range(batch_size):
        tmp_nb_x = lms_pred_nb_x[idx][reverse_index1, reverse_index2].view(num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[idx][reverse_index1, reverse_index2].view(num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x[idx].unsqueeze(1), tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y[idx].unsqueeze(1), tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)
        # Transform coordinates from [0, 1] to [-1, 1] for FaceXFormer compatibility
        lms_pred_merge = 2 * lms_pred_merge - 1
        landmark_output.append(lms_pred_merge)
    
    return torch.stack(landmark_output)

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (30)
    (0.0, 330.0, -65.0),        # Chin (8)
    (-225.0, -170.0, -135.0),     # Left eye left corner (36)
    (225.0, -170.0, -135.0),      # Right eye right corner (45)
    (-150.0, 150.0, -125.0),    # Left mouth corner (48)
    (150.0, 150.0, -125.0)      # Right mouth corner (54)
], dtype="double")
landmark_indices = [30, 8, 36, 45, 48, 54]
def get_angles(landmarks, img_size):
    width = height = img_size
    image_points = []
    for i in landmark_indices:
        x_norm = landmarks[f'x{i}']
        y_norm = landmarks[f'y{i}']
        x_pixel = (x_norm + 1) / 2 * width
        y_pixel = (y_norm + 1) / 2 * height
        image_points.append((x_pixel, y_pixel))
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(mat)
    yaw = eulerAngles[1][0]
    pitch = eulerAngles[0][0]
    roll = eulerAngles[2][0]
    return yaw, pitch, roll

def get_landmarks(input_path, input_name, output_path, model_path, model_type, device="cuda:0", batch_size=32, meanface_path=None):
    os.makedirs(output_path, exist_ok=True)
    model, extra_data = load_model(model_type, model_path, device, meanface_path)
    transforms_image = get_transforms(model_type)
    
    dataset = ImageDataset(input_path, transform=transforms_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    results = []
    for batch_images, batch_filenames in tqdm(dataloader, desc="Processing batches"):
        batch_images = batch_images.to(device)
        
        if model_type == "facexformer":
            landmark_output = get_landmarks_facexformer(model, batch_images, device)
        elif model_type == "pipnet":
            landmark_output = get_landmarks_pipnet(
                model, batch_images,
                extra_data["reverse_index1"],
                extra_data["reverse_index2"],
                extra_data["max_len"]
            )
        for filename, landmarks in zip(batch_filenames, landmark_output):
            landmarks = landmarks.cpu().numpy().flatten()
            result = {'filename': filename}
            for i in range(68):
                result[f'x{i}'] = landmarks[2 * i]
                result[f'y{i}'] = landmarks[2 * i + 1]
            yaw, pitch, roll = get_angles(result, batch_images.shape[2])
            result['yaw'] = yaw
            result['pitch'] = pitch
            result['roll'] = roll
            results.append(result)
    
    df = pd.DataFrame(results)
    output_file = os.path.join(output_path, f"lm_{input_name}_{model_type}.h5")
    df.to_hdf(output_file, key='landmarks', mode='w', format='table')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input images")
    parser.add_argument("--input_name", help="Input dataset name")
    parser.add_argument("--output", required=True, help="Path to save aligned faces (df dir)")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--model_type", required=True, choices=["facexformer", "pipnet"], help="Model type")
    parser.add_argument("--meanface_path", required=True, help="Path to meanface.txt (for pipnet)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device to run model on")
    args = parser.parse_args()
    
    mlflow.set_experiment("landmark")
    with mlflow.start_run():
        mlflow.log_param("input_path", args.input)
        mlflow.log_param("input_name", args.input_name)
        mlflow.log_param("output_path", args.output)
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("meanface_path", args.meanface_path)
        mlflow.log_param("device", args.device)
        mlflow.log_param("batch_size", args.batch_size)
        
        get_landmarks(args.input, args.input_name, args.output, args.model, args.model_type, args.device, args.batch_size, args.meanface_path)