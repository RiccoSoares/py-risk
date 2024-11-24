import unittest
from coevolution import coevolution
import risk
from risk.orders import *
from risk.rand import rand_move
from risk.game_types import MapState
from risk import custom_maps
from risk.nn import *

class TestCoevolution(unittest.TestCase):
    def setUp(self):
        mapstruct = custom_maps.create_banana_map()
        self.mapstate = mapstruct.randState()
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
            p1_individual = OrderList.from_gene(
                self.coevolution.population1[i], self.mapstate.mapstruct, self.player1
            )
            p2_individual = OrderList.from_gene(
                self.coevolution.population2[i], self.mapstate.mapstruct, self.player2
            )
            self.assertIsInstance(p1_individual, OrderList)
            self.assertIsInstance(p2_individual, OrderList)

if __name__ == '__main__':
    unittest.main()