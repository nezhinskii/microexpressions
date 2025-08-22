import argparse
import os
import mlflow
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .facexformer import FaceXFormer

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

def get_landmarks(input_path, output_path, model_path, device="cuda:0", batch_size=32):
    os.makedirs(output_path, exist_ok=True)
    
    model = FaceXFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    model.eval()
    
    transforms_image = transforms.Compose([
        transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageDataset(input_path, transform=transforms_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(device)
        task = torch.tensor([1]).repeat(len(batch_images)).to(device)
        labels = {
            "segmentation": torch.zeros([len(batch_images), 224, 224]).to(device),
            "lnm_seg": torch.zeros([len(batch_images), 5, 2]).to(device),
            "landmark": torch.zeros([len(batch_images), 68, 2]).to(device),
            "headpose": torch.zeros([len(batch_images), 3]).to(device),
            "attribute": torch.zeros([len(batch_images), 40]).to(device),
            "a_g_e": torch.zeros([len(batch_images), 3]).to(device),
            'visibility': torch.zeros([len(batch_images), 29]).to(device)
        }

        with torch.no_grad():
            landmark_output, _, _, _, _, _, _, _ = model(batch_images, labels, task)

        for filename, landmarks in zip(batch_filenames, landmark_output):
            landmarks = landmarks.cpu().numpy().flatten()  # [68, 2] -> [136]
            result = {'filename': filename}
            for i in range(68):
                result[f'x{i}'] = landmarks[2 * i]
                result[f'y{i}'] = landmarks[2 * i + 1]
            results.append(result)
            
    df = pd.DataFrame(results)
    output_file = os.path.join(output_path, "landmarks.h5")
    df.to_hdf(output_file, key='landmarks', mode='w', format='table')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input images")
    parser.add_argument("--output", required=True, help="Path to save aligned faces")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device to run model on")
    args = parser.parse_args()
    
    mlflow.set_experiment("face_alignment")
    with mlflow.start_run():
        mlflow.log_param("input_path", args.input)
        mlflow.log_param("output_path", args.output)
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("device", args.device)
        mlflow.log_param("batch_size", args.batch_size)
        
        get_landmarks(args.input, args.output, args.model, args.device, args.batch_size)