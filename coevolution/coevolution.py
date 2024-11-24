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
     initialize_pops_with_gnn= False,
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
        self.eval_table = np.zeros((populations_size, populations_size))
        self.eval_table_ready = [False]
        self.population1 = []
        self.population2 = []
        self.population1_fitnesses = []
        self.population2_fitnesses = []
        self.initialize_pops_with_gnn = initialize_pops_with_gnn
    
    def evolve(self):
        initialize_populations()
        for generation in range(self.generations):

            if timeout is not np.inf and time() > timeout:
                break

            evaluate_populations()
            apply_crossover()
            mutate_populations()
            evaluate_populations()
            apply_elitism()

    def initialize_populations(self):
        self.population1 = [
            rand_move(self.mapstate, self.player1).to_gene(self.mapstruct) 
            for _ in range(self.populations_size)
        ]

        self.population2 = [
            rand_move(self.mapstate, self.player2).to_gene(self.mapstruct) 
            for _ in range(self.populations_size)
        ]

    def evaluate_populations(self):
        self.eval_table[:] = 0
        for i in range(self.populations_size):
            for j in range(self.populations_size):
                pop1_individual = OrderList.from_gene(self.population1[i], self.mapstruct, self.player1)
                pop2_individual = OrderList.from_gene(self.population2[j], self.mapstruct, self.player2)
                self.eval_table[i, j] = fitness(pop1_individual, pop2_individual)

    def apply_crossover(self):
        pass

    def mutate_populations(self):
        pass

    def evaluate_board_position(self, mapstate):
        return 1

    def fitness(self, pop1_individual, pop2_individual):
        resulting_board = (pop1_individual | pop2_individual)(self.mapstate)
        return evaluate_board_position(resulting_board)

    def consult_fitness(self, individual):
        pass

    def apply_elitism(self):
        pass
