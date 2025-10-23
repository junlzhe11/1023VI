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
topTen_dir = './data/topTen/'

# Measure the similarity scores between query feature and gallery features.
# You could also use other metrics to measure the similarity scores between features.
def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def retrival_ranklist(queryIndex):
    results = []  # Store retrieval results for both features
    similarity_dict = {}  # Moved outside to be shared between both features
    
    for feat_idx in range(2):  # Process feat0 and feat1
        query_feat_path = os.path.join(query_feat_dir, f'query{queryIndex}feat{feat_idx}.npy')
        
        # Check if feature file exists
        if not os.path.exists(query_feat_path):
            print(f"Warning: Feature file {query_feat_path} not found, skipping...")
            continue
            
        try:
            # Load query feature
            query_feat = np.load(query_feat_path)
            
            # Iterate through gallery feature files
            for gallery_feat_file in os.listdir(gallery_feat_dir):
                # Use gallery_feat_file instead of cnt
                gallery_feat_path = os.path.join(gallery_feat_dir, gallery_feat_file)
                
                # Check if gallery feature file exists
                if not os.path.exists(gallery_feat_path):
                    continue
                    
                try:
                    gallery_feat = np.load(gallery_feat_path)
                    gallery_idx = gallery_feat_file.split('.')[0] + '.jpg'
                    sim = similarity(query_feat, gallery_feat)
                    
                    if feat_idx == 0:
                        # First feature, store directly
                        similarity_dict[gallery_idx] = sim
                    else:
                        # Second feature, take maximum value
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
    
    # Sort after processing all features
    if similarity_dict:
        sorted_similarity = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_similarity
    else:
        return []

def visulization(retrieved, query):
    # Create larger canvas to display 1 query + 10 results
    plt.figure(figsize=(20, 8))
    
    # First subplot: Query image
    plt.subplot(2, 6, 1)
    plt.title('Query Image')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    plt.axis('off')
    
    # Display top 10 retrieval results
    for i in range(min(10, len(retrieved))):
        img_path = './data/gallery/' + retrieved[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        
        plt.subplot(2, 6, i + 2)
        plt.title(f'Top {i+1}\nScore: {retrieved[i][1]:.3f}')
        plt.imshow(img_rgb)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Automatically save results
    os.makedirs(topTen_dir, exist_ok=True)
    
    query_index = os.path.basename(query).split('.')[0].replace('query', '')
    save_path = os.path.join(topTen_dir, f'query{query_index}_retrieval_top10.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved: {save_path}")
    plt.close()

if __name__ == '__main__':
    # Create a list to store all ranklist lines
    all_ranklists = []
    
    for queryIndex in range(50):
        ranklist = retrival_ranklist(queryIndex) # retrieve top matching images in the gallery.
        best_ten = ranklist[:10]
        print(f"Query {queryIndex} - Top 10: {best_ten}")
        
        # Create ranklist line for this query - include ALL results
        if ranklist:
            # Extract numeric IDs from gallery filenames (remove .jpg extension)
            rank_ids = []
            for gallery_file, score in ranklist:
                img_id = gallery_file.replace('.jpg', '')
                rank_ids.append(img_id)
            
            # Format: "Q1: 7 12 214 350 ..." with ALL IDs, no omission
            ranklist_line = f"Q{queryIndex + 1}: " + " ".join(rank_ids)
        else:
            ranklist_line = f"Q{queryIndex + 1}:"
        
        all_ranklists.append(ranklist_line)
        
        query_path = os.path.join(query_dir, f'{queryIndex}.jpg')
        visulization(best_ten, query_path) # Visualize the retrieval results
    
    # Save all ranklists to a single file with complete data
    os.makedirs('./data/ranklists/', exist_ok=True)
    ranklist_file_path = './data/ranklists/ranklist_complete.txt'
    
    with open(ranklist_file_path, 'w') as f:
        for line in all_ranklists:
            f.write(line + '\n')
    
    print(f"Complete ranklists saved to: {ranklist_file_path}")
    print(f"Total queries processed: {len(all_ranklists)}")
    print(f"Each line contains all retrieval results for that query")