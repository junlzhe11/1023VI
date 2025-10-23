import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

query_dir = './data/query/'
query_feat_dir = './data/query_feat/'
query_txt_dir = './data/query_txt/'
query_box_dir = './data/query_box' 
query_cropped_dir = './data/cropped_query/'
gallery_dir = './data/gallery/'
gallery_feat_dir = './data/gallery_feature/'

def query_crops(query_path, txt_path, queryIndex):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] # Convert BGR to RGB
    txt = np.loadtxt(txt_path)     # Load the coordinates of the bounding box

    crops = []  # Array to store all crops
    save_path = os.path.join(query_cropped_dir, f'query{queryIndex}crop0.jpg')
    # Handle single bounding box (1D array)
    if txt.ndim == 1:
        if len(txt) == 4:
            # Single bounding box
            x, y, w, h = txt
            crop = query_img[int(y):int(y + h), int(x):int(x + w), :]
            cv2.imwrite(save_path, crop[:,:,::-1])  # Save the cropped region
            crops.append(crop)
        else:
            # Multiple boxes in one row (shouldn't happen with 1-2 boxes)
            print(f"Unexpected number of coordinates: {len(txt)}")

    # Handle multiple bounding boxes (2D array)
    elif txt.ndim == 2:
        for i, bbox in enumerate(txt):
            x, y, w, h = bbox[:4]
            crop = query_img[int(y):int(y + h), int(x):int(x + w), :]

            # Save each crop with appropriate filename
            if len(txt) == 1:
                # Single box - use original save_path
                cv2.imwrite(save_path, crop[:,:,::-1])
            else:
                # Multiple boxes - add index to filename
                individual_save_path = os.path.join(query_cropped_dir, f'query{queryIndex}_crop{i+1}.jpg')
                cv2.imwrite(individual_save_path, crop[:,:,::-1])

            crops.append(crop)

    print(f"Cropped {len(crops)} instances from {query_path}")
    return crops

def resnet_extraction(img, featsave_path, model=None, resize=(224, 224)):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_transform = resnet_transform(img_rgb)
    img_transform = torch.unsqueeze(img_transform, 0).to(device)  # Use automatic device detection
    
    # If no model is provided, create a new ResNet
    if model is None:
        model = models.resnet101(pretrained=True)
    
    model