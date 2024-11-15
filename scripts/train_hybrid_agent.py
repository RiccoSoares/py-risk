import argparse
from nn.models import Model5  # Ensure proper imports
from agents.hybrid_agent import HybridAlphaZeroGA

def main(args):
    policy_value_net = Model5()
    agent = HybridAlphaZeroGA(policy_value_net)
    agent.train(args.iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()
    main(args)