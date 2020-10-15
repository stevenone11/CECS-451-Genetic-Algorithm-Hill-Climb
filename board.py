import random
import numpy as np
import time

class Node:
    def __init__(self, i, j):
        self.setRow(i)
        self.setCol(j)
    def setRow(self, num):
        self.row = num
    def setCol(self, num):
        self.col = num
    def getRow(self):
        return self.row
    def getCol(self):
        return self.col

# the board class
class Board:
    def __init__(self, n):
        self.n_queen = n
        self.map = [[0 for j in range(n)] for i in range(n)]
        self.fit = 0
        self.nonFit = 0
        self.queens = []
    
    #creates the table
        for i in range(self.n_queen):
            j = random.randint(0, self.n_queen - 1)
            self.map[i][j] = 1
            self.queens.append(Node(i, j))

    # the non fitness portion of the board simply calculates how many non attacking pairs exist
    def nonFitness(self):        
        for i in range(self.n_queen):
            for j in range(self.n_queen):
                if self.map[i][j] == 1:
                    for k in range(1, self.n_queen - i):
                        if self.map[i + k][j] == 0:
                            self.nonFit += 1
                        if j - k >= 0 and self.map[i + k][j - k] == 0:
                            self.nonFit += 1
                        if j + k < self.n_queen and self.map[i + k][j + k] == 0:
                            self.nonFit += 1
    
    def fitness(self):        
        for i in range(self.n_queen):
            for j in range(self.n_queen):
                if self.map[i][j] == 1:
                    for k in range(1, self.n_queen - i):
                        if self.map[i + k][j] == 1:
                            self.fit += 1
                        if j - k >= 0 and self.map[i + k][j - k] == 1:
                            self.fit += 1
                        if j + k < self.n_queen and self.map[i + k][j + k] == 1:
                            self.fit += 1

    def show(self):
        print(np.matrix(self.map))
        print("Fitness: ",  self.fit)

    def flip(self, i, j):
        if self.map[i][j] == 0:
            self.map[i][j] = 1
        else:
            self.map[i][j] = 0

    def get_map(self):
        return self.map
    
    def get_fit(self):
        return self.fit

    def get_nonFit(self):
        return self.nonFit

    def copy(self, other):
        other.queens = []
        for i in range(self.n_queen):
            other.queens.append(self.queens[i])
            for j in range(self.n_queen):
                other.map[i][j] = self.map[i][j]
        other.fitness()
        
    def setQueen(self, i, j):
        self.queens[i].setCol(j)
        self.map[i][j] = 1

    def removeQueen(self, i, j):
        self.map[i][j] = 0