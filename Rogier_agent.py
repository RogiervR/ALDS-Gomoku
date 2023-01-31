import random, time, copy
import numpy as np
from pygame.locals import KEYUP,QUIT,MOUSEBUTTONUP,K_ESCAPE
import gomoku
from GmUtils import GmUtils
from gomoku import Board, Move, GameState, valid_moves, pretty_board


class Node():
    def __init__(self, state, last_move, valid_moves , parent_node=None ):
        self.state = state
        self.parent_node: Node= parent_node
        self.children: list[Node] = [] 
        self.last_move = last_move
        self.valid_moves = valid_moves
        self.moves_left = self.valid_moves[:]
        self.N = 0
        self.Q = 0


    def UCT(self):
        return (self.Q / self.N) + (0.71 * np.sqrt((2 * np.log(self.N)) / self.N))

    def get_QN(self):
        return self.Q / self.N


#The Big Oh is O(mkI/C), waar m de hoeveelheid random children is, k de hoeveelheid parallele zoekresultaten, I de hoeveelheid iteraties en C de hoeveelheid beschikbare Cores)
class Rogier_player:
    def __init__(self, black_=True):
        self.black = black_
        self.root_node = None
    
        self.max_move_time_ns   = 0
        self.start_time_ns      = 0
        self.Amount_MCTS_loops = 7

    def new_game(self, black_: bool):
        """At the start of each new game you will be notified by the competition.
        this method has a boolean parameter that informs your agent whether you
        will play black or white.
        """
        self.black = black_

    def id(self) -> str:
        """Please return a string here that uniquely identifies your submission e.g., "name (student_id)" """
        return "Rogier van Rooijen (1786347)"

    #De Big Oh van deze functie is O(1), omdat je een keer door de gegeven leaf node heen gaat om te kijken welke child de beste UTC waarde heeft.
    def FindSpotToExpand(self, nleaf):
        if gomoku.check_win(nleaf.state[0], nleaf.last_move) or not nleaf.valid_moves:
                return nleaf
        if nleaf.moves_left:
            rand_move = random.choice(nleaf.moves_left)
            nleaf.moves_left.remove(rand_move)

            state_backup = copy.deepcopy(nleaf.state)
            valid ,is_win , new_state = gomoku.move(state_backup, rand_move)
            new_node = Node(state = nleaf.state[0], last_move= rand_move, valid_moves=new_state)

            valid_moves_list = [i for i in nleaf.valid_moves if i != rand_move]
            new_node = Node(state = new_state, last_move= rand_move, valid_moves=valid_moves_list , parent_node=nleaf)
            nleaf.children.append(new_node)
            return new_node

        best_child = nleaf.children[0]
        best_child_value = nleaf.children[0].get_QN()
        for child in nleaf.children:
                child_value = child.get_QN()
                if best_child_value < child_value:
                    best_child = child
                    best_child_value = child_value
        return self.FindSpotToExpand(best_child)
    

    #De Big Oh van deze functie is O(1), omdat je een keer door alle valid moves van de gegeven leaf node looped en kijkt of je met die gegeven move wint.
    def Rollout(self , nleaf):
        win = gomoku.check_win(nleaf.state[0], nleaf.last_move)
        state_backup = copy.deepcopy(nleaf.state)
        moves_left = nleaf.moves_left[:]
        while not win:
            if not moves_left: #Gelijk
                return 0.5  

            random_move = random.choice(moves_left)
            gomoku.move(state_backup, random_move)
            moves_left.remove(random_move)

        if (nleaf.state[1] % 2 != 0 and self.black) or (nleaf.state[1] % 2 == 0 and not self.black):
            return 0
        else: 
            return 1 
                
        

    #De Big Oh van deze functie is O(1), omdat je maar een keer door de nleaf looped om te checken of die niet None is, waarbij je de Q en N van de leaf update.
    def BackUpValue(self , val , nleaf):
        while nleaf is not None:
            nleaf.N += 1
            if (nleaf.state[1] % 2 != 0 and self.black) or (nleaf.state[1] % 2 == 0 and not self.black):
                nleaf.Q -= val
            else:
                nleaf.Q += val

            nleaf = nleaf.parent_node


    #De Big Oh van deze functie is O(n), omdat je een keer per loop expand en een rollout doet per nieuwe node wat maakt dat je een lineair verband krijgt.
    def move(self, state: GameState, L_move: Move, max_time_to_move: int = 1000) -> Move:
            startTime = time.time_ns()
            valid_moves = GmUtils.getValidMoves(state[0], state[1])
            nroot = Node(state, L_move, valid_moves)
            while(((time.time_ns() - startTime) / 1000000) < max_time_to_move):
                nleaf = self.FindSpotToExpand(nroot)
                for i in range(self.Amount_MCTS_loops):
                    val = self.Rollout(nleaf)
                    self.BackUpValue( val , nleaf)
            best_child = nroot.children[0]
            best_child_value = nroot.children[0].get_QN()
            
            for child in nroot.children:
                child_value = child.get_QN()
                if best_child_value < child_value:
                    best_child = child
                    best_child_value = child_value
            
            return best_child.last_move
