import risk
from risk.orders import *
from risk.rand import rand_move
import numpy as np
from risk.game_types import MapState

class Individual:
    def __init__(self, genes, index):
        self.genes = genes
        self.fitness = None
        self.index = index

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness

class Coevolution:
    def __init__(self,
     mapstate: MapState,
     player1: int, 
     player2: int,
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
        self.relational_fitness_table = np.zeros((populations_size, populations_size))
        self.population1 = []
        self.population2 = []
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
        for i in range(self.populations_size):
            self.population1.append(
                Individual(rand_move(self.mapstate, self.player1).to_gene(self.mapstruct), i)
            )

            self.population2.append(
                Individual(rand_move(self.mapstate, self.player2).to_gene(self.mapstruct), i)
            )

    def evaluate_populations(self):
        self.relational_fitness_table[:] = 0
        for i in range(self.populations_size):
            for j in range(self.populations_size):
                self.relational_fitness_table[i, j] = self.relational_fitness(
                    self.population1[i], self.population2[j]
                )

        for i in range(self.populations_size):
            self.population1[i].fitness = np.mean(self.relational_fitness_table[i,:])
            self.population2[i].fitness = np.mean(self.relational_fitness_table[:,i])

    def relational_fitness(self, pop1_individual, pop2_individual):
        pop1_order = OrderList.from_gene(pop1_individual.genes, self.mapstruct, self.player1)
        pop2_order = OrderList.from_gene(pop2_individual.genes, self.mapstruct, self.player2)

        resulting_board = (pop1_order | pop2_order)(self.mapstate)
        return self.evaluate_board_position(resulting_board)

    def evaluate_board_position(self, mapstate):
        #will get the board position and pass to the neural network model
        return 1

    def mutate_populations(self):
        pass

    def apply_elitism(self):
        pass

    def apply_crossover(self):
        pass
