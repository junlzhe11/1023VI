# Retrieve the most similar images by measuring the similarity between features.
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

query_dir = './data/query/'
query_feat_dir = './data/query_feat/'
query_txt_dir = './data/query_txt/'
query_box_dir = './data/query_box' 
query_cropped_dir = './data/cropped_query/'
gallery_dir = './data/gallery/'
gallery_feat_dir = './data/gallery_feature/'

# Measure the similarity scores between query feature and gallery features.
# You could also use other metrics to measure the similarity scores between features.
def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def retrival_idx(queryIndex):
    results = []  # 儲存兩種特徵的檢索結果
    similarity_dict = {}  # 移到外面，讓兩個特徵可以共用
    
    for feat_idx in range(2):  # 處理 feat0 和 feat1
        query_feat_path = os.path.join(query_feat_dir, f'query{queryIndex}feat{feat_idx}.npy')
        
        # 檢查特徵文件是否存在
        if not os.path.exists(query_feat_path):
            print(f"Warning: Feature file {query_feat_path} not found, skipping...")
            continue
            
        try:
            # 載入查詢特徵
            query_feat = np.load(query_feat_path)
            
            # 遍歷 gallery 特徵文件
            for gallery_feat_file in os.listdir(gallery_feat_dir):
                # 修正：使用 gallery_feat_file 而不是 cnt
                gallery_feat_path = os.path.join(gallery_feat_dir, gallery_feat_file)
                
                # 檢查 gallery 特徵文件是否存在
                if not os.path.exists(gallery_feat_path):
                    continue
                    
                try:
                    gallery_feat = np.load(gallery_feat_path)
                    gallery_idx = gallery_feat_file.split('.')[0] + '.jpg'
                    sim = similarity(query_feat, gallery_feat)
                    
                    if feat_idx == 0:
                        # 第一次特徵，直接存入
                        similarity_dict[gallery_idx] = sim
                    else:
                        # 第二次特徵，取最大值
                        if gallery_idx in similarity_dict:
                            similarity_dict[gallery_idx] = max(sim, similarity_dict[gallery_idx])
                        else:
                            similarity_dict[gallery_idx] = sim
                            
                except Exception as e:
                    print(f"Error loading gallery feature {gallery_feat_path}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing feature {query_feat_path}: {e}")
            continue
    
    # 在所有特徵處理完後排序
    if similarity_dict:
        sorted_similarity = sorted(similarity_dict.items(), key=lambda item: item[1])
        best_ten = sorted_similarity[-10:]  # 取相似度最高的5個
        return best_ten
    else:
        return []

def visulization(retrieved, query):
    # 創建更大的畫布來顯示 1 個查詢 + 10 個結果
    plt.figure(figsize=(20, 8))
    
    # 第一個子圖：查詢圖像
    plt.subplot(2, 6, 1)  # 改為 2x6 的網格
    plt.title('Query Image')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    plt.axis('off')
    
    # 顯示前10個檢索結果
    for i in range(min(10, len(retrieved))):  # 防止 retrieved 少於10個
        img_path = './data/gallery/' + retrieved[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        
        # 計算子圖位置
        plt.subplot(2, 6, i + 2)  # 第2到第11個位置
        plt.title(f'Top {i+1}\nScore: {retrieved[i][1]:.3f}')
        plt.imshow(img_rgb)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    # save result

if __name__ == '__main__':
    for queryIndex in range(50):
        best_ten = retrival_idx(queryIndex) # retrieve top 5 matching images in the gallery.
        print(best_ten)
        best_ten.reverse()
        query_path = os.path.join(query_dir, f'{queryIndex}.jpg')
        visulization(best_ten, query_path) # Visualize the retrieval results

