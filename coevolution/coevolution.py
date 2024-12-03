import risk
from risk.orders import *
from risk.rand import rand_move
import numpy as np
from risk.game_types import MapState
from time import time

class Coevolution:
    def __init__(self,
     mapstate: MapState,
     player1: int, 
     player2: int,
     gnn_model, 
     populations_size= 20, 
     generations= 10,  
     crossover_rate= 0.2, 
     tournament_size= 3, 
     elitism= 4,
     mutation_rate= 0.05,
     mutation_percent_genes = 0.05,
     initialize_pops_with_gnn= False,
     timeout= np.inf):

        self.mapstate = mapstate
        self.mapstruct = mapstate.mapstruct
        self.player1 = player1
        self.player2 = player2
        self.gnn_model = gnn_model
        self.populations_size = populations_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.mutation_percent_genes = mutation_percent_genes
        self.population = []
        self.best_individual = None
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
        
        self.evaluate_populations()
        self.population1.sort(reverse=True)
        self.population2.sort(reverse=True)

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
            self.population2[i].fitness = - np.mean(self.relational_fitness_table[:,i])

    def relational_fitness(self, pop1_individual, pop2_individual):
        pop1_order = OrderList.from_gene(pop1_individual.genes, self.mapstruct, self.player1)
        pop2_order = OrderList.from_gene(pop2_individual.genes, self.mapstruct, self.player2)

        resulting_board = (pop1_order | pop2_order)(self.mapstate)
        return self.evaluate_board_position(resulting_board)

    def evaluate_board_position(self, mapstate):
        if mapstate.winner() is not None:
            return 1 if mapstate.winner() == self.player1 else -1

        graph_features, global_features, edges = mapstate.to_tensor(self.player1, self.player2)
        v, _ = self.gnn_model(graph_features, global_features, edges, [])
            
        return v * 0.5

    def mutate_populations(self):
        for i in range(self.populations_size):
            if np.random.rand() < self.mutation_rate:
                self.mutate(self.population1[i], self.player1)
                self.mutate(self.population2[i], self.player2)
        
    def mutate(self, individual, player):
        offspring_genes = individual.genes.copy()
        for i in range(len(offspring_genes)):
            if np.random.rand() < self.mutation_percent_genes:
                offspring_genes[i] = np.random.randint(0, 10)

        return self.correct_mutation(offspring_genes, player)

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

        self.population1 = offspring1
        self.population2 = offspring2
        
    def crossover_population(self, population):
        offspring = []
        while len(offspring) < self.populations_size:
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            child1, child2 = self.crossover(parent1, parent2)
            offspring.extend([child1, child2])
        return offspring[:self.populations_size]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1.genes) - 1)

        child1_genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
        child2_genes = np.concatenate((parent2.genes[:crossover_point], parent1.genes[crossover_point:]))

        child1 = Individual(child1_genes, parent1.index)
        child2 = Individual(child2_genes, parent2.index)
        return child1, child2

    def correct_mutation(self, offset_genes, player):
        edges = self.mapstruct.edgeLabels()

        # Remove attacks from unowned territories
        for (src, dst), index in edges.items():
            if self.mapstate.owner[src] != player:
                offset_genes[index] = 0
        # Remove excess deployments
        while offset_genes[:len(self.mapstate)].sum() > self.mapstate.income(player):
            j = np.random.choice(np.where(offset_genes[:len(self.mapstate)] > 0)[0])
            offset_genes[j] -= 1
        # Add missing deployments
        while offset_genes[:len(self.mapstate)].sum() < self.mapstate.income(player):
            j = np.random.choice(np.where(self.mapstate.owner == player)[0])
            offset_genes[j] += 1
            
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