import cv2
import numpy as np

def extract_cell(i,j,scaled_img):
    size = scaled_img.shape[0]
    cell_size = size/15
    cell_img = scaled_img[int(i*cell_size):int((i+1)*cell_size), int(j*cell_size):int((j+1)*cell_size)]
    return cell_img

def predict_letter(cell_img, extractor, model):
    grey = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
    processed_img = extractor(grey,return_tensors="pt")
    logits = model(**processed_img)
    predicted_label = logits['logits'].argmax(-1).item()
    if logits['logits'].max().item() < 5:
        return ''
    return model.config.id2label[predicted_label]

def color_cluster(mat, i, j, c):
    n_rows,n_cols = mat.shape
    
    mat[i,j] = c
    
    for x in range(max(0,i-1), min(n_rows-1, i+1)+1):
        for y in range(max(0,j-1), min(n_cols-1, j+1)+1):
            if mat[x,y] == 0 and (x != i or y != j):
                mat = color_cluster(mat, x, y, c)
    return mat
    
def get_clusters(grey):
    res = grey.copy()
    c = 1
    n_rows,n_cols = grey.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if res[i,j]==0:
                res = color_cluster(res, i, j, c)
                c = c + 1

    res[res==255]=0
    dic = {}
    big_clust, big_clust_size = 0, 0 
    for t in range(1,c):

        dic[t] = np.count_nonzero(res == t)
        if dic[t] > big_clust_size:
            big_clust, big_clust_size = t, dic[t]
    return res, dic, big_clust
        
def process_cell_cluster(cell_img): 
    threshold = 20
    
    is_letter = False
    res, dic, big_clust = get_clusters(cell_img)
    out = res.copy()
    if big_clust != 0 and dic[big_clust] > threshold:
        is_letter = True
    out[out != big_clust] = 255
    out[out == big_clust] = 0
    return is_letter, out
    
                
        

    

