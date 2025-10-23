#HOW TO INSTALL ANNACONDA: https://www.youtube.com/watch?v=YJC6ldI3hWk
#WHAT IS IMAGENET DATABASE: https://www.youtube.com/watch?v=gogV2wKKF_8
import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from ultralytics import YOLO

query_dir = './data/query/'
query_feat_dir = './data/query_feat/'
query_txt_dir = './data/query_txt/'
query_box_dir = './data/query_box' 
query_cropped_dir = './data/cropped_query/'
gallery_dir = './data/gallery/'
gallery_feat_dir = './data/gallery_feature/'

def query_crops(query_path, txt_path, queryIndex):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box

    crops = []  # Array to store all crops
    save_path = os.path.join(query_cropped_dir, f'query{queryIndex}crop0.jpg')
    # Handle single bounding box (1D array)
    if txt.ndim == 1:
        if len(txt) == 4:
            # Single bounding box
            x, y, w, h = txt
            crop = query_img[int(y):int(y + h), int(x):int(x + w), :]
            cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
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

def yolo_11_extraction(img, featsave_path):
    yolo = YOLO("./data/model/yolo11n.pt").eval()
    net= yolo.model
    features = {}
    def hook_fn(module, input, output):
        features['feat'] = output
    target_layer = net.model[22]
    target_layer.register_forward_hook(hook_fn)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        _ = net(img_tensor)
    feat_map = features['feat']  # tensor shape: [1, C, H, W]
    feats = torch.mean(feat_map, dim=[2,3])
    feats_np = feats.cpu().detach().numpy() # convert tensor to numpy
    np.save(featsave_path, feats_np) # save the feature

# Note that I feed the whole image into the pretrained vgg11 model to extract the feature, which will lead to a poor retrieval performance.
# To extract more fine-grained features, you could preprocess the gallery images by cropping them using windows with different sizes and shapes.
# Hint: opencv provides some off-the-shelf tools for image segmentation.
def feat_extractor_gallery(gallery_dir, gallery_feat_dir):
    # 檢查設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for img_file in tqdm(os.listdir(gallery_dir)):
        img = cv2.imread(os.path.join(gallery_dir, img_file))
        img = img[:,:,::-1] #bgr2rgb
        img_resize = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC) # resize the image
        featsave_path = os.path.join(gallery_feat_dir, img_file.split('.')[0]+'.npy')
        yolo_11_extraction(img_resize, featsave_path)

def feat_extractor_query():
    # 創建目錄（如果不存在）
    os.makedirs(query_cropped_dir, exist_ok=True)
    os.makedirs(query_feat_dir, exist_ok=True)
    os.makedirs(query_box_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 處理50張查詢圖片 (0.jpg 到 49.jpg)
    for queryIndex in tqdm(range(50), desc="Processing query images"):
        # 構建文件路徑
        query_path = os.path.join(query_dir, f'{queryIndex}.jpg')
        query_txt_path = os.path.join(query_txt_dir, f'{queryIndex}.txt')

        # 檢查文件是否存在
        if not os.path.exists(query_path):
            print(f"Query image {query_path} not found, skipping...")
            continue
        if not os.path.exists(query_txt_path):
            print(f"Query text {query_txt_path} not found, skipping...")
            continue

        try:
            # 裁剪和特徵提取
            crops = query_crops(query_path, query_txt_path, queryIndex)

            # 檢查crops是否為空
            if crops is None or len(crops) == 0:
                print(f"No crops found for query image {queryIndex}")
                continue
            for cropIndex in range(len(crops)):
                crop = crops[cropIndex]
                crop_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
                featsave_path = os.path.join(query_feat_dir, f'query{queryIndex}feat{cropIndex}.npy')
                yolo_11_extraction(crop_resize, featsave_path)
                print(f"Successfully processed image {queryIndex} crop {cropIndex}")      
        except Exception as e:
            print(f"Error processing query image {queryIndex}: {e}")

def main():
    feat_extractor_query()
    feat_extractor_gallery(gallery_dir, gallery_feat_dir)

if __name__=='__main__':
    main()