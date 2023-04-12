import cv2
import numpy as np

# def get_contour(img):
#     grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (7, 7), 0)
#     edges = cv2.Canny(blur, 50, 100, apertureSize=3)
#     kernel = np.ones((5, 5), np.uint8)
#     dilation = cv2.dilate(edges, kernel, iterations=2)
#     contour_img, contours, hierarchy = cv2.findContours(
#         dilation,
#         cv2.RETR_TREE,
#         cv2.CHAIN_APPROX_SIMPLE)
#     cnt = max(contours, key=cv2.contourArea)
#     contour_img = img.copy()
#     contour_img = cv2.drawContours(contour_img, [cnt], 0, (0, 255, 0), 3)
#     return contour_img, cnt

#If the below function does not work try the function above
def get_contour(img):
    """This function returns the contours of the scrabble board"""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 100, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=2)
    contours,contour_img = cv2.findContours(
        dilation,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    contour_img = img.copy()
    contour_img = cv2.drawContours(contour_img, [cnt], 0, (0, 255, 0), 3)
    return contour_img, cnt

def get_corners(cnt):
    """returns the corners of the scrabble board"""
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return [x[0] for x in approx]

def rescale_image(corners, img, out_size = 400):
    """"returns a new rescale image where the corners of the 
    board are the corners of the image"""
    tl, tr, bl, br = None, None, None, None
    m_x = sum([tp[0]/4 for tp in corners])
    m_y = sum([tp[1]/4 for tp in corners])
    for tp in corners:
        if tp[0] > m_x and tp[1]>m_y:
            br = tp
            nbr = [out_size, out_size]
        elif tp[0] < m_x and tp[1]>m_y:
            bl = tp
            nbl = [0, out_size]
        elif tp[0] < m_x and tp[1]<m_y:
            tl = tp
            ntl = [0, 0]
        elif tp[0] > m_x and tp[1]<m_y:
            tr = tp
            ntr = [out_size, 0]
    input_pts = np.float32([tl, tr, br, bl])
    output_pts = np.float32([ntl, ntr, nbr, nbl])        
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(img,M,(out_size, out_size),flags=cv2.INTER_LINEAR)
    return out


            
        