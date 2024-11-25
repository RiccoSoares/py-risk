import unittest
from unittest.mock import patch
from coevolution import coevolution
import risk
from risk.orders import *
from risk.rand import rand_move
from risk.game_types import MapState
from risk import custom_maps
from risk.nn import *

class TestCoevolution(unittest.TestCase):
    def setUp(self):
        self.mapstruct = custom_maps.create_banana_map()
        self.mapstate = MapState(
            armies=np.array([10, 5, 7, 8, 3, 6, 9, 4, 2, 1, 8, 7]),
            owner=np.array([1, 1, 2, 2, 0, 1, 2, 1, 1, 2, 0, 2]),
            mapstruct=self.mapstruct
        )
        self.player1 = 1
        self.player2 = 2
        self.gnn_model = Model12()
        self.populations_size = 10
        self.generations = 5
        self.mutation_rate = 0.04
        self.crossover_rate = 0.7
        self.tournament_size = 3
        self.elitism = 1
        self.timeout = float('inf')
        self.coevolution = coevolution.Coevolution(
            self.mapstate, 
            self.player1, 
            self.player2, 
            self.gnn_model, 
            self.populations_size, 
            self.generations, 
            self.mutation_rate, 
            self.crossover_rate, 
            self.tournament_size, 
            self.elitism, 
            self.timeout
        )

    def test_initialize_populations(self):
        self.coevolution.initialize_populations()
        self.assertEqual(len(self.coevolution.population1), self.populations_size)
        self.assertEqual(len(self.coevolution.population2), self.populations_size)
        for i in range(self.populations_size):
            p1_individual_genes = OrderList.from_gene(
                self.coevolution.population1[i].genes, self.mapstate.mapstruct, self.player1
            )
            p2_individual_genes = OrderList.from_gene(
                self.coevolution.population2[i].genes, self.mapstate.mapstruct, self.player2
            )
            self.assertIsInstance(p1_individual_genes, OrderList)
            self.assertIsInstance(p2_individual_genes, OrderList)
            self.assertIsInstance(self.coevolution.population1[i], coevolution.Individual)
            self.assertIsInstance(self.coevolution.population2[i], coevolution.Individual)

    def test_evaluate_populations(self):
        self.coevolution.initialize_populations()
        self.coevolution.evaluate_populations()
        self.assertEqual(self.coevolution.relational_fitness_table.shape,
         (self.populations_size, self.populations_size)
        )
        for i in range(self.populations_size):
            for j in range(self.populations_size):
                self.assertIsInstance(self.coevolution.relational_fitness_table[i][j], float)
    
    @patch.object(coevolution.Coevolution, 'evaluate_board_position', return_value=1.0)
    def test_evaluate_board_position_mocked(self, mock_evaluate_board_position):
        self.coevolution.initialize_populations()
        self.coevolution.evaluate_populations()

        for i in range(self.populations_size):
            self.assertEqual(self.coevolution.population1[i].fitness, 1.0)
            self.assertEqual(self.coevolution.population2[i].fitness, 1.0)

        mock_evaluate_board_position.assert_called()

    def test_define_elites(self):
        self.coevolution.initialize_populations()

        for ind in self.coevolution.population1:
            ind.fitness = ind.index
        for ind in self.coevolution.population2:
            ind.fitness = ind.index

        self.coevolution.define_elites()
        self.assertEqual(self.coevolution.population1_elite[0].fitness, self.populations_size - 1)
        self.assertEqual(self.coevolution.population2_elite[0].fitness, self.populations_size - 1)

    def test_mutate(self):
        # Define valid orders for the initial map state
        valid_orders = OrderList([
            DeployOrder(self.player1, 0, 5),  # Deploy 5 armies to territory 0
            DeployOrder(self.player1, 1, 3),  # Deploy 3 armies to territory 1
            AttackTransferOrder(self.player1, 0, 1, 2)  # Attack from 0 to 1 with 2 armies
        ])

        # Convert valid orders to genes
        initial_genes = valid_orders.to_gene(self.mapstruct)
        
        # Create an individual with these initial genes
        individual = coevolution.Individual(initial_genes, 0)

        # Mutate the individual
        self.coevolution.mutate(individual)
        
        # Validate mutation by checking if modified within defined rules
        mutated_genes = individual.genes
        
        # Ensure no negative values exist
        self.assertTrue((mutated_genes >= 0).all())
        
        # Ensure no over-deployment based on player1 income
        self.assertLessEqual(mutated_genes[:len(self.mapstate)].sum(), self.mapstate.income(self.player1))

        # Ensure that attacks are valid from owned territories
        for (src, dst), index in self.mapstruct.edgeLabels().items():
            if src != dst and self.mapstate.owner[src] != self.player1:
                self.assertEqual(mutated_genes[index], 0)
                
        # Ensure deployments sum to the income of player1
        self.assertEqual(mutated_genes[:len(self.mapstate)].sum(), self.mapstate.income(self.player1))


if __name__ == '__main__':
    unittest.main()