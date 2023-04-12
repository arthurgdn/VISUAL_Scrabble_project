from Solver.solver import Solver
from Solver.dawg import *

def test_solver():
    word_rack = ["Z","L","R","A","A","V","S"]
    board = [['E', '', '', '', '', '', '', 'F', '', '', '', '', '', '', ''],
 ['', '', '', '', '', '', 'V', 'A', 'T', '', '', '', '', '', ''],
 ['', '', '', '', '', '', '', 'N', 'O', '', '', '', '', '', '_'],
 ['', '', '', '', '', '', '', '', 'L', 'N', '', '', '', '', '_'],
 ['', '', '', '', '', '', '', '', 'L', 'J', '', '', '', '', '_'],
 ['', 'J', '', '', '', '', '', '', 'E', 'X', '', '', '', '', ''],
 ['Q', 'L', '', '', '', '', '', '', 'T', '', '', '', '', '', ''],
 ['U', 'N', 'T', 'L', 'E', '', 'D', 'U', 'E', '', '', '', '', '', '_'],
 ['A', '', '', '', 'G', 'O', 'O', '', 'D', 'O', '', '', '', '', '_'],
 ['Y', 'O', '', 'T', 'O', 'Y', '', '', '', 'U', '', 'T', 'E', 'A', ''],
 ['', '', '', 'R', '', '', '', 'F', '', 'N', '', 'A', '', '', '_'],
 ['', '', '', 'U', '', '', '', 'E', '', 'C', '', 'B', '', '', '_'],
 ['', '', 'W', 'E', '', '', '', 'W', 'H', 'E', 'A', 'L', 'S', '', '_'],
 ['', 'P', 'O', '', '', 'V', 'L', 'E', '', '', '', 'E', '', '', '_'],
 ['Z', 'A', 'H', '', '', '', '', 'R', '', '', '_', 'T', 'O', 'T', 'L']]
    solver = Solver(board)
    print(solver.get_best_word(word_rack))

test_solver()