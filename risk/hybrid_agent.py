import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .pygad import GA  # Ensure pygad is setup correctly
from .game_manager import GameManager  # Assuming game_manager handles the game logic

class HybridAlphaZeroGA:
    def __init__(self, policy_value_net, population_size=100, generations=50, evolution_interval=10):
        self.policy_value_net = policy_value_net
        self.population_size = population_size
        self.generations = generations
        self.evolution_interval = evolution_interval
        self.replay_buffer = ReplayBuffer()

    def train(self, n_iterations):
        for t in range(n_iterations):
            self.play_games()
            if t % self.evolution_interval == 0:
                self.evolve()
            self.update_networks()

    def play_games(self):
        def play_games(self):
        game_manager = GameManager() 
        mcts = MCTS(mapstate=game_manager.get_initial_mapstate(), p1=1, p2=2, model=self.policy_value_net)
        game_data = mcts.play(game_manager.get_initial_mapstate())
        for data in game_data:
            self.replay_buffer.push(data)

    def evolve(self):
        # Initialize populations using policy predictions
        initial_population = self.initialize_population()
        ga = GA(population=initial_population, num_generations=self.generations, 
                num_parents_mating=self.population_size // 2, fitness_func=self.fitness_func)
        evolved_population = ga.run()
        self.apply_evolved_population(evolved_population)

    def initialize_population(self):
        # Use outputs from the policy network to create an initial population of moves
        initial_population = []
        for _ in range(self.population_size):
            # Assuming policy_value_net outputs policy logits that can be turned into probabilities
            policy_logits, _ = self.policy_value_net(game_state)  # Replace game_state with correct input
            probs = F.softmax(policy_logits, dim=-1)
            move = np.random.choice(len(probs), p=probs.detach().numpy())
            initial_population.append(move)
        return initial_population

    def fitness_func(self, individual):
        # Evaluate the fitness of moves in the population
        game_state_after_move = self.simulate_game_step(individual)
        _, value = self.policy_value_net(game_state_after_move)
        return value.item()

    def apply_evolved_population(self, evolved_population):
        # Integrate the evolved population into the learning process
        for move in evolved_population:
            # Apply these moves and record the outcomes in the replay buffer
            game_state = self.simulate_game_step(move)
            self.replay_buffer.push(game_state)  # Assuming game_state contains the necessary data format

    def update_networks(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        self.train_network(batch)

    def train_network(self, batch):
        optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        optimizer.step()
    
    def compute_loss(self, batch):
        # Computing loss for the policy and value network
        states, actions, rewards, next_states = zip(*batch)
        logits, values = self.policy_value_net(states)
        # Compute policy loss and value loss
        policy_loss = F.cross_entropy(logits, actions)
        value_loss = F.mse_loss(values, rewards)
        return policy_loss + value_loss

    def simulate_game_step(self, move):
        # Use the game manager to simulate the game step
        new_game_state = GameManager().apply_move(move)  # Ensure apply_move simulates the move
        return new_game_state

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def push(self, data):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
