#%%
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Image, display
import PIL.Image
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch

from ultralytics import YOLO

QUERY_PATH = "./data/query"
BOX_PATH="./data/query_txt"
GALLERY_PATH = "./data/gallery"

def display_image(cv2_image):
    image = cv2_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(image))

# 读取query的crop后的图片
query_img = cv2.imread(os.path.join(QUERY_PATH,"query.jpg"))
with open(os.path.join(BOX_PATH,"query.txt"),"r", encoding="utf-8") as f:
    line = f.readline()
    x,y,w,h = tuple(map(int, list(line.split(' '))))

display_image(query_img)

# 绘制croped box
croped_box_img = cv2.rectangle(query_img.copy(), (x,y),(x+w,y+h), (0,0,255), 1)

display_image(croped_box_img)

# 读取crop的坐标
query_croped = query_img[y:y+h,x:x+w,:]

display_image(query_croped)

# 获取特征值

#%%
# resized_croped = cv2.resize(query_croped, (224,224), interpolation=cv2.INTER_CUBIC)
# display_image(resized_croped)

model = models.resnet101(pretrained=True).cuda()


def vgg_features(model, origin_cv2_image, resize=None):
    cp = origin_cv2_image.copy()
    if resize is not None:
        cp = cv2.resize(cp, resize, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(cp, cv2.COLOR_BGR2RGB)
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_transform = resnet_transform(img) 
    img_transform = torch.unsqueeze(img_transform, 0) 
    model.eval()
    with torch.no_grad():
        feature = model(img_transform.cuda())
    return feature.cpu()

query_feature = vgg_features(model, query_croped, (224,224))

# %%
from tqdm.auto import tqdm
def cossim(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

# 读取gallery中的图片
rank_list = []
for i in tqdm(range(22404)):
    if not os.path.exists(os.path.join(GALLERY_PATH, f"{i}.jpg")):
        continue
    gallery_img = cv2.imread(os.path.join(GALLERY_PATH, f"{i}.jpg"))
    gh,gw,gc = gallery_img.shape
    # 默认box框是小于图片的
    gallery_feature = vgg_features(model, gallery_img, resize=(224,224))
    score = cossim(query_feature, gallery_feature )
    # max_area = (-1,-1,-1,-1)
    # print(i,gh,gw,w,h)
    # for j in range(0, gh-h, max(int(gh/h),1)):
    #     for k in range(0,gw-w, max(int(gw/w),1)):
    #         area_image = gallery_img[j:j+h, k:k+w]
    #         area_feature = vgg_features(model, area_image)
    #         simscore = cossim(query_feature, area_feature)
    #         if simscore>score:
    #             score = simscore
    #             max_area = (k,j,w,h)

    rank_list.append((i, score))

rank_list.sort(key=lambda x:x[1], reverse=True)

def visulization(retrived, query):
    plt.figure(figsize=(24,16))
    plt.subplot(2, 6, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(10):
        img_path = './data/gallery/' + f"{retrived[i][0]}.jpg"
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(2, 6, i+2)
        plt.title(f"top{i}:{retrived[i][1]}")
        plt.imshow(img_rgb)
    plt.show()

visulization(rank_list, os.path.join(QUERY_PATH, "query.jpg"))


