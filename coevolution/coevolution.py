import risk
from risk.orders import *
from risk.rand import rand_move
import numpy as np

class Coevolution:
    def __init__(self, mapstate,
     player1, 
     player2,
     gnn_model, 
     populations_size= 10, 
     generations= 5, 
     mutation_rate= 0.04, 
     crossover_rate= 0.7, 
     tournament_size= 3, 
     elitism= 1,
     timeout= np.inf):

        self.mapstate = mapstate
        self.mapstruct = mapstate.mapstruct
        self.player1 = player1
        self.player2 = player2
        self.gnn_model = gnn_model
        self.populations_size = populations_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.best_generation = 0
        self.fitnesses = []
        self.eval_table = np.zeros((populations_size, populations_size))
        self.eval_table_ready = [False]
        self.population1 = []
        self.population2 = []
    
    def evolve(self):
        pass

    def initialize_populations(self):
        self.population1 = [rand_move(self.mapstate, self.player1).to_gene(self.mapstruct) for _ in range(self.populations_size)]
        self.population2 = [rand_move(self.mapstate, self.player2).to_gene(self.mapstruct) for _ in range(self.populations_size)]

    def fitness(self, individual):
        pass
