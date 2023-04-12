import pickle
from Solver.board import Board
from Solver.dawg import *

class Solver:
    def __init__(self, board, vocab_path="data/vocab_dawg.pickle"):
        #Load the lexicon with the DAWG structure
        to_load = open(vocab_path, "rb")
        root = pickle.load(to_load)
        to_load.close()
        self.vocab_dawg = root
        #Init a game board
        self.board = Board(self.vocab_dawg, board)

    # scan all tiles on board in both transposed and non-transposed state, find best move
    def get_best_word(self, word_rack):
        word_rack = [l.upper() for l in word_rack]
        self.board.modified = False
        if self.board.is_empty():
            #If the board is empty we use a specific function to get the best first word
            return self.get_start_word(word_rack)
        
        # clear out cross-check lists before adding new words
        self.board.update_cross_checks()

        # reset word variables to clear out words from previous turns
        self.board.best_word = ""
        self.board.highest_score = 0
        self.board.best_row = 0
        self.board.best_col = 0

        #Check all possible words in both directions and keep the one with highest score
        transposed = False
        for row in range(0, 15):
            for col in range(0, 15):
                curr_square = self.board.board[row][col]
                if curr_square.letter and (not self.board.board[row][col - 1].letter):
                    prev_best_score = self.board.highest_score
                    self.board.get_all_words(row + 1, col + 1, word_rack)
                    if self.board.highest_score > prev_best_score:
                        self.board.best_row = row
                        self.board.best_col = col

        self.board.transpose()
        for row in range(0, 15):
            for col in range(0, 15):
                curr_square = self.board.board[row][col]
                if curr_square.letter and (not self.board.board[row][col - 1].letter):
                    prev_best_score = self.board.highest_score
                    self.board.get_all_words(row + 1, col + 1, word_rack)
                    if self.board.highest_score > prev_best_score:
                        transposed = True
                        self.board.best_row = row
                        self.board.best_col = col

        # Don't try to insert word if we couldn't find one
        if not self.board.best_word:
            self.board.transpose()
            return "Couldn't find any word to place on grid", None, None
        

        # Insert the new word on the board and update the score
        if transposed:
            self.board.insert_word(self.board.best_row + 1, self.board.best_col + 1 - self.board.dist_from_anchor, self.board.best_word)
            self.board.transpose()
        else:
            self.board.transpose()
            self.board.insert_word(self.board.best_row + 1, self.board.best_col + 1 - self.board.dist_from_anchor, self.board.best_word)

        self.board.word_score_dict[self.board.best_word] = self.board.highest_score
        self.board.print_board()

        if not self.board.modified:
            return "Couldn't find any word to place on grid", None, None
        
        return self.board.best_word, self.board.best_row + 1, self.board.best_col + 1
    
    def get_start_word(self, word_rack):
        # board symmetrical at start so just always play the start move horizontally
        # try every letter in rack as possible anchor square
        self.board.best_row = 7
        self.board.best_col = 8
        for i, letter in enumerate(word_rack):
            potential_square = self.board.board[7][8]
            temp_rack = word_rack[:i] + word_rack[i + 1:]
            potential_square.letter = letter
            self.board.left_part(self.vocab_dawg, 7, 8, temp_rack, "", [], 6, 1)

        # reset anchor square spot to blank after trying all combinations
        self.board.board[7][8].letter = None
        self.board.insert_word(self.board.best_row + 1, self.board.best_col + 1 - self.board.dist_from_anchor, self.board.best_word)
        self.board.board[7][8].modifier = ""
        self.board.word_score_dict[self.board.best_word] = self.board.highest_score
        self.board.print_board()
        return self.board.best_word, self.board.best_row + 1, self.board.best_col + 1