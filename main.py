from Vision.rescale import get_contour
from Vision.rescale import get_corners
from Vision.rescale import rescale_image
from Vision.get_grid import threshold_grid
from Vision.get_grid import proceed_grid
import cv2
from transformers import AutoFeatureExtractor
import torch
from Solver.solver import Solver
from Solver.dawg import *

### All process ###

# load image
filename = 'data/board1.jpeg'
img = cv2.imread(filename,1)

# obtain contours
contour_img, cnt = get_contour(img)

# get corners of contours
corners = get_corners(cnt)

# rescale image
scaled_img = rescale_image(corners, img)

# threshold img
th_img = threshold_grid(scaled_img)

# extractor and model 
extractor = AutoFeatureExtractor.from_pretrained("pittawat/vit-base-letter")
model = torch.load('model/vit-ft.pt')

# extract grid
grid = proceed_grid(th_img, extractor, model)
solver = Solver(grid)
solver.board.print_board()
word_rack = list(input('Quelles sont vos lettres ?'))
print(solver.get_best_word(word_rack))




