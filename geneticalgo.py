# steven centeno
# Shoraj Manandhar

import random
import numpy as np
import time

# we will have a state class that inculdes each staes board, encoded string, fitness value (of 
# non attacking pairs) as well as its normal value
# the encoded string, fitness and nomal value all have setters and getters for their 
# value to allow for easy access and change to the variables, the board must be accessed manually
class State:
    def __init__(self, board):
        self.board = board

    def setEncodedStr(self, enc):
        self.enc = enc
    
    def getEncodedStr(self):
        return self.enc
        
    def setFit(self, fit):
        self.fit = int(fit)

    def getFit(self):
        return int(self.fit)

    def setNormVal(self, normVal):
        self.normVal = normVal

    def getNormVal(self):
        return self.normVal


# the board class
class Board:
    def __init__(self, n):
        self.n_queen = n
        self.map = [[0 for j in range(n)] for i in range(n)]
        self.fit = 0
        self.nonFit = 0
    
    #creates the table
        for i in range(self.n_queen):
            j = random.randint(0, self.n_queen - 1)
            self.map[i][j] = 1

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


# initial input should look something like geneticAlgo([], 0, 8, 5)
# runs the geneti algorithm given the states
# input - list of state object, # of times repeated(generation), n # of states, q # of queens
# output - board pinted, the running time, # of restarts
def geneticAlgo(state, restart, n, q):

    # if the # of states in the list is 0, encode the states
    if len(state) == 0:
        for i in range(n):
            boardN = Board(q) # create board with q queens
            boardN.nonFitness()
            encodedState = encode(boardN, q) # get the encoded string
            nState = State(boardN) # initialize state with the board 
            nState.setEncodedStr(encodedState) # give the state its encoded string
            nState.setFit(boardN.get_nonFit()) # since we are searching for non fitness values
            state.append(nState) # give empty list the nth state with its board and encoded string

    # estimates which state has the highest non attacking pair and returns randomly
    # states equal to the states given, allowing for repeated states
    newStateList = selection(state) # will first randomly select between the states based on their normal value percentage
    pairAndCross = pairing(newStateList) # will then pair, swap and mutate elements in the encoded strings of the states
    lastStates = decode(pairAndCross, q, restart) # will decode the changes in the string onto the board and if the solution is found, exit

    if(len(lastStates) == 0): # exits the program when the solution is found
        return
    
    return geneticAlgo(lastStates, restart + 1, n, q) # this will recursively call


# takes in a board class and convets it to a string
# input - board class, n # of queens
# output- the encoded string representing the board
def encode(board, q):
    myMap = board.get_map()
    encodedStr = ""

    for i in range(q):
        for j in range(q):
            if myMap[i][j] == 1:
                encodedStr += str(j)

    return encodedStr


# gets a list of states, gets their fitness value, and computes each
# normal value (probability of selection) then selects randomly between
# the states up to the number of states that exist in the array (the states
# can have repeating states selected)
# input - the array holding the states
# output - the states that were randomly selected between the given input
def selection(lState):
    
    totalVal = 0 # keep track of the total fitness
    
    # goes through each state to add its fitness to total 
    for i in range(len(lState)):
        totalVal += int(lState[i].getFit())

    # calculates the normal value per state 
    for i in range(len(lState)):
        curFit= int(lState[i].getFit())
        lState[i].setNormVal(curFit/totalVal)
    
    # selects randomly one of the states and saves it to a new list
    # repeats this up to the number of states in the original list
    numOfRandomSelec = len(lState)
    newStates = []
    for j in range(numOfRandomSelec): # get n number of statesto give to new list
        randomSelec = random.uniform(0, 1) # random number from 0 - 1
        totalnormVal = 0 # the total current normal value saved while looping
        alreadySelected = False 

        # how we get the random number from the list is as such:
        # get the random generated number ex: 0.7 and loop through the list
        # as it loops through the list, it will get each elements normal value
        # since the list will look somethat like that [[.02],[.2],[.3],[.3],[.18]]
        # we need to save the total normal value as we go through the list
        # so first element .02 is less than .7, we will continue, next element is 
        # .2 which is added to the total to get .22 still less than .7 continue, 
        # .3 + .22 = .52 < .7, continue so now .3 + .52 = .82 which is greater than .7, this means it is
        # the number we selected  
        for i in range(len(lState)):
            currentNorm = lState[i].getNormVal()
            totalnormVal += currentNorm # keeping track of the current total normal value

            if randomSelec < totalnormVal and not(alreadySelected): 
                newStates.append(lState[i]) # saves the new state that was randomly selected
                alreadySelected = True
    
    return newStates

# takes in an array of the states and does the following:
# 1. shuffles the array so that 2 nearby states can be randomly paried with one another
# 2. swaps elements in the paired encoded states by placing them in the swap function
# 3. mutates a random element fom the encoded state twice for more diverse solutions
# input - all the states in an array
# output - all the states in the array having been paired, corssovered and mutated
def pairing(states):
    random.shuffle(states) # shuffle the states in order to randomly pair the arrays selected
    sizeBoard = states[0].board.n_queen # gets the size of q queens for later in the code
    position = 0

    # loops through the states to pair the states 2 at a time
    while position < len(states):
        encodedStr1 = states[position].getEncodedStr() # get the encoded of the first and second state
        encodedStr2 = states[position + 1].getEncodedStr()
        str1, str2 = crossover(encodedStr1, encodedStr2) # cross over the pairs
        str1 = mutation(str1, sizeBoard) # now randomly mutate the pairs individually
        str2 = mutation(str2, sizeBoard) 
        str1 = mutation(str1, sizeBoard) # now we again mutate one element from the pairs for more diverse solutions
        str2 = mutation(str2, sizeBoard) 
        states[position].setEncodedStr(str1) # overwrite the changed encoded strings to teh newer versions
        states[position + 1].setEncodedStr(str2)
        position += 2
    
    return states


# gets two encoded strings and crosses them over through a randomly selected pivot
# input - two encoded strings
# output- the same two encoded strings having been crossed over
def crossover(str1, str2):
    randIndex = random.randint(1,len(str1) - 1) # randomly get an index to cross over the arrays
    nextS1 = str1[0:randIndex] + str2[randIndex:] # crosses over the arrays with one another
    nextS2 = str2[0:randIndex] + str1[randIndex:]

    return nextS1, nextS2


# takes in an encoded string and mutates one element in the string by a 95% chance
# input- the encoded string and size of the board
# output- the mutated string
def mutation(enc, n):

    # needs a probability of mutating before deciding to mutate
    mutProb = random.uniform(0,1)
    if mutProb < 0.95: # 95% chance of mutating
        rand_index = random.randint(1,n) # gets a random position to mutate
        randomNum = str(random.randint(0,n - 1))
        mutated = enc[0:rand_index-1] + randomNum + enc[rand_index:]
        return mutated

    return enc

# takes the states and decodes the encoded states into the board
# input- an array of states, the size of the board, the number of generations passed(restarts)
# output- the board decoded with all the variables in the board and state updated
def decode(states, n, restarts):

    # will loop through all the states to both decode the strings to their proper board 
    # and reinitialize the fitness and non attacking pairs for recursive use
    for i in range(len(states)):
        encodedStr = states[i].getEncodedStr() # gets the encoded string for the current state
        mapEncodedStr(states[i], encodedStr, n) 
        states[i].board.fit = 0 # makes the fitness 0 to rest the fitness value
        states[i].board.nonFit = 0
        states[i].board.fitness() # calls fitness to get the current fitness value
        states[i].board.nonFitness()
        Fit = states[i].board.get_fit() # saves the fitness to a variable
        nonFit = states[i].board.get_nonFit()
        states[i].setFit(nonFit) # saves the fitness value to the state

        if Fit == 0: # detcts if one of the states found a solution
            states[i].board.show()
            print("non Attacking Pairs:{0:}\n# of generations:{1:}".format(nonFit, restarts))
            return []

    return states

# maps the encoded string to the current state
# input - the current state, the encoded string, the size of the board
# output -  the decoded board
def mapEncodedStr(currState, encodedStr, n):
    currState.board.map = [[0 for j in range(n)] for i in range(n)] # reinitializes board to all 0's
    
    # loop through the encoded string and do the following:
    # 1. get the character of the current position, this is the column
    # 2. the current position is the row of the board
    # 3. get the board and place the queens in its row and column
    for i in range(len(encodedStr)):
        column = int(encodedStr[i]) # gets the value for the column
        currState.board.map[i][column] = 1 # places the queen in its proper row and column

if __name__ == '__main__':
    runningtime = time.time() # keeps track of the current time
    geneticAlgo([], 0, 8, 5)
    endtime = time.time() # gets the time once the program ended
    print("time spent running: " + str((endtime - runningtime) * 1000))# prints the running time   