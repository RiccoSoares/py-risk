import risk
from risk.orders import *
from risk.rand import rand_move
import numpy as np
from risk.game_types import MapState
from time import time
from risk.data_loader import *
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

class Individual:
    def __init__(self, genes, index, fitness=None):
        self.genes = genes
        self.fitness = fitness
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
     initialize_populations_with_policy= False, 
     populations_size= 30, 
     generations= 15,  
     crossover_rate= 0.2, 
     tournament_size= 3, 
     elitism= 5,
     mutation_rate= 0.2,
     mutation_percent_genes = 0.05,
     timeout= 12):

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
        self.initialize_populations_with_policy = initialize_populations_with_policy
        self.timeout = timeout
        self.start_time = time()
    
    def evolve(self):
        self.initialize_populations()
        for generation in range(self.generations):

            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.evaluate_populations()
            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.define_elites()
            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.apply_crossover()
            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.mutate_populations()
            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.evaluate_populations()
            if self.timeout is not np.inf and time() - self.start_time > self.timeout:
                break

            self.apply_elitism()
        
        self.evaluate_populations()
        self.population1.sort(reverse=True)
        self.population2.sort(reverse=True)

    def get_populations_with_policy(self):
        pop1_moves, pop2_moves = [], []

        for i in range(self.populations_size*3):
            pop1_moves.append(rand_move(self.mapstate, self.player1))
            pop2_moves.append(rand_move(self.mapstate, self.player2))
        
        pop1_data = self.prep_policy_value_data(self.mapstate, self.mapstruct, pop1_moves, self.player1, self.player2)
        pop2_data = self.prep_policy_value_data(self.mapstate, self.mapstruct, pop2_moves, self.player2, self.player1)
        pop1_loader = DataLoader([pop1_data], batch_size=1, shuffle=False)
        pop2_loader = DataLoader([pop2_data], batch_size=1, shuffle=False)

        for batch in pop1_loader:
            _, policy1 = self.gnn_model(batch)
        for batch in pop2_loader:
            _, policy2 = self.gnn_model(batch)

        policy1 = policy1[0].tolist()
        policy2 = policy2[0].tolist()

        for i in range(self.populations_size*3):
            self.population1.append(Individual(pop1_moves[i].to_gene(self.mapstruct), i, policy1[i]))
            self.population2.append(Individual(pop2_moves[i].to_gene(self.mapstruct), i, policy2[i]))

        self.population1.sort(reverse=True)
        self.population2.sort(reverse=True)
        self.population1 = self.population1[:self.populations_size]
        self.population2 = self.population2[:self.populations_size]

    def initialize_populations(self):
        if self.initialize_populations_with_policy:
            self.get_populations_with_policy()
            return

        for i in range(self.populations_size):
            self.population1.append(
                Individual(rand_move(self.mapstate, self.player1).to_gene(self.mapstruct), i)
            )
            self.population2.append(
                Individual(rand_move(self.mapstate, self.player2).to_gene(self.mapstruct), i)
            )

    def evaluate_populations(self):
        self.relational_fitness_table[:] = 0
        boards_to_evaluate = []
        boards_indices = []
        winners = []

        for i in range(self.populations_size):
            for j in range(self.populations_size):
                pop1_order = OrderList.from_gene(self.population1[i].genes, self.mapstruct, self.player1)
                pop2_order = OrderList.from_gene(self.population2[j].genes, self.mapstruct, self.player2)
                resulting_board = (pop1_order | pop2_order)(self.mapstate)
                
                w = resulting_board.winner()
                if w is not None:
                    winners.append(True)
                    fit = 1 if w == self.player1 else -1
                    self.relational_fitness_table[i, j] = fit

                else:
                    winners.append(False)
                    boards_to_evaluate.append(resulting_board)
                    boards_indices.append((i, j))

        boards_data = []
        for board, won in zip(boards_to_evaluate, winners):
            if not won:
                rand_moves = [rand_move(board, self.player1) for i in range(5)]
                prep = self.prep_policy_value_data(board, self.mapstruct, rand_moves, self.player1, self.player2)
                boards_data.append(prep)
        
        if len(boards_data) > 0:
            data_loader = DataLoader(boards_data, batch_size=200, shuffle=False)
            values = []
            for batch in data_loader:
                v, _ = self.gnn_model(batch)
                values.append(v)
            values = torch.cat(values, dim=0)

        else:
            values = torch.tensor([])

        eval_index = 0
        for idx, (i, j) in enumerate(boards_indices):
            if not winners[idx]:
                fit = values[eval_index].item()
                self.relational_fitness_table[i, j] = fit
                eval_index += 1

        for i in range(self.populations_size):
            self.population1[i].fitness = np.mean(self.relational_fitness_table[i,:])
            self.population2[i].fitness = - np.mean(self.relational_fitness_table[:,i])

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

    def prep_policy_value_data(self, state, mapstruct, moves, player, opponent):
        x1, x2, edges = state.to_tensor(player, opponent)
        graph_features, _, _ = state.to_tensor(player, opponent, full=False)
        i1, i2 = state.income(player), state.income(opponent)
        assert torch_geometric.utils.is_undirected(edges)

        mask, nodes, values, b_edges, b_mapping = mapstruct.bonusTensorAlt()

        z = torch.zeros(values.size(), dtype=torch.long)
        z.index_add_(0, mask, torch.ones(mask.size(), dtype=torch.long))

        orders = moves
        order_data = build_order_data(orders, state, x1)
        return StateData(
            map=mapstruct.id,
            num_nodes=len(mapstruct),
            num_bonuses=len(values),
            num_moves=len(orders),
            graph_data=x1,
            global_data=x2,
            graph_features=graph_features,
            graph_edges=edges,
            bonus_edges=b_edges,
            bonus_batch=mask,
            bonus_nodes=nodes,
            bonus_values=values,
            bonus_values_normed=values / z,
            bonus_mapping=b_mapping,
            income=torch.tensor([i1, i2]).view(1, -1),
            total_armies=graph_features[:,2:].sum(dim=0).view(1,-1),
            edge_index=edges,
            **order_data,
        )
            