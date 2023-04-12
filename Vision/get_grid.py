import cv2
from tqdm import tqdm
from Vision.letter_detection import extract_cell
from Vision.letter_detection import process_cell_cluster
from Vision.letter_detection import predict_letter

def threshold_grid(scaled_img):
    filter_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2HSV)
    th_img = cv2.adaptiveThreshold(filter_img[:, :, 2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,151,90)
    return th_img

def proceed_grid(th_img, extractor, model): 
    res = []
    for i in tqdm(range(15)):       
        row = []
        for j in range(15):
            
            cell_img = extract_cell(i,j,th_img)
            test, p_img = process_cell_cluster(cell_img)
            if test:
                prediction = predict_letter(p_img, extractor, model)
            else:
                prediction = ''
            row.append(prediction)
        res.append(row)
    return res