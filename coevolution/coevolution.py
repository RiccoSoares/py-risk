import risk
from risk.orders import *
from risk.rand import rand_move
import numpy as np
from risk.game_types import MapState
from time import time

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
        self.relational_fitness_table = np.zeros((populations_size, populations_size))
        self.population1 = []
        self.population2 = []
        self.population1_elite = []
        self.population2_elite = []
        self.initialize_pops_with_gnn = initialize_pops_with_gnn
        self.timeout = timeout
        self.start_time = time()
    
    def evolve(self):
        self.initialize_populations()
        for generation in range(self.generations):

            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.evaluate_populations()
            self.define_elites()
            self.apply_crossover()
            self.mutate_populations()
            self.evaluate_populations()
            self.apply_elitism()

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
        for population in [self.population1, self.population2]:
            for individual in population:
                if np.random.rand() < self.mutation_rate:
                    self.mutate(individual)
        
    def mutate(self, individual):
        edges = self.mapstruct.edgeLabels()
        offspring = individual.genes.copy()

        offspring[offspring < 0] = 0

        attacks_from = {src: np.zeros(len(edges), dtype=bool) for src in range(len(self.mapstate))}
        for (src, dst), index in edges.items():
            if src != dst:
                attacks_from[src][index] = True

        player = self.mapstate.owner[individual.index]
        for (src, dst), index in edges.items():
            if self.mapstate.owner[src] != player:
                offspring[index] = 0

        deployment_sum = offspring[:len(self.mapstate)].sum()
        if deployment_sum > self.mapstate.income(player):
            diff = deployment_sum - self.mapstate.income(player)
            while diff > 0:
                i = np.random.choice(np.where(offspring[:len(self.mapstate)] > 0)[0])
                offspring[i] -= 1
                deployment_sum -= 1
                diff -= 1

        elif deployment_sum < self.mapstate.income(player):
            diff = self.mapstate.income(player) - deployment_sum
            while diff > 0:
                i = np.random.choice(np.where(self.mapstate.owner == player)[0])
                offspring[i] += 1
                deployment_sum += 1
                diff -= 1

        individual.genes = offspring

    def apply_elitism(self):
        self.population1.sort(reverse=True)
        self.population2.sort(reverse=True)

        pop1_best_offsprings = self.population1[:self.populations_size - len(self.population1_elite)]
        pop2_best_offsprings = self.population2[:self.populations_size - len(self.population2_elite)]

        self.population1 = self.population1_elite + pop1_best_offsprings
        self.population2 = self.population2_elite + pop2_best_offsprings

    def define_elites(self):
        self.population1.sort(reverse=True)
        self.population2.sort(reverse=True)

        self.population1_elite = self.population1[:self.elitism]
        self.population2_elite = self.population2[:self.elitism]

    def apply_crossover(self):
        offspring1 = self.crossover_population(self.population1)
        offspring2 = self.crossover_population(self.population2)

        self.population1 = self.population1_elite + offspring1[:self.populations_size - len(self.population1_elite)]
        self.population2 = self.population2_elite + offspring2[:self.populations_size - len(self.population2_elite)]
        
    def crossover_population(self, population):
        offspring = []
        while len(offspring) < self.populations_size - len(self.population1_elite):
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            child1, child2 = self.crossover(parent1, parent2)
            offspring.extend([child1, child2])

        return offspring[:self.populations_size - len(self.population1_elite)]

    def crossover(self, parent1, parent2):
        #yet to implement crossover logic
        return parent1, parent2
