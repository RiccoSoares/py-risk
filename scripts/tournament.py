import datetime as dt
import argparse
import json
import pickle
import os
from distutils.util import strtobool

import risk
import risk.custom_maps as custom_maps
from risk.nn import Model5
from risk.replay_buffer import ReplayBuffer

try:
    from risk.nn import *
except ImportError:
    pass

def __main__(args):
    gnn = Model15()
    weights_pth = f'model-weights/it2.pkl'

    with open(weights_pth, 'rb') as f:
        gnn.load_state_dict(pickle.load(f))

    gnn.eval()

    guided_mcts_config = {
        "type": "MCTS",
        "iters": 300,
        "model": gnn,
        "initialize_populations_with_policy": False,
        "name": "Guided MCTS",
    }

    baseline_mcts_config = {
        "type": "MCTS",
        "iters": 300,
        "model": None,
        "initialize_populations_with_policy": False,
        "name": "Baseline MCTS",
    }

    ga_config = {
        "type": "Genetic",
        "iters": 150,
        "model": gnn,
        "initialize_populations_with_policy": False,
        "name": "Genetic Algorithm",
    }

    policy_initialized_ga_config = {
        "type": "Genetic",
        "iters": 150,
        "model": gnn,
        "initialize_populations_with_policy": True,
        "name": "Policy Initialized Genetic Algorithm",
    }

    matchups = [(baseline_mcts_config, ga_config),
                (baseline_mcts_config, policy_initialized_ga_config),
                (baseline_mcts_config, guided_mcts_config),
                (guided_mcts_config, ga_config),
                (guided_mcts_config, policy_initialized_ga_config),
                (ga_config, policy_initialized_ga_config),]
    
    maps = [custom_maps.create_owl_island_map(), custom_maps.create_simple_map(), custom_maps.create_banana_map()]

    buffer_paths = [f'replay-buffer/tournament/owl_island.pkl', f'replay-buffer/tournament/simple.pkl', f'replay-buffer/tournament/banana.pkl']

    for matchup in matchups:
        print(f"\n\nStarting matchup between {matchup[0]['name']} and {matchup[1]['name']}\n\n")

        bot1 = risk.mcts_helper.model_builder(matchup[0]["type"])(
                None, 1, 2, matchup[0]["model"],
                iters=matchup[0]["iters"],
                max_depth=args.max_depth_1,
                trust_policy=args.policy_trust_1,
                moves_to_consider=args.moves_consider_1,
                timeout=12,
                exploration=args.exploration_1,
                cache_opponent_moves=args.cache_opponent_moves_1,
                obj_rand=args.obj_rand_1,
                alpha=args.alpha_1,
                pop_size=args.pop_size_1,
                mirror_model=args.mirror_model_1,
                initialize_populations_with_policy = matchup[0]["initialize_populations_with_policy"],
        )

        bot2 = risk.mcts_helper.model_builder(matchup[0]["type"])(
                None, 2, 1, matchup[1]["model"],
                iters=matchup[1]["iters"],
                max_depth=args.max_depth_2,
                trust_policy=args.policy_trust_2,
                moves_to_consider=args.moves_consider_2,
                timeout=12,
                exploration=args.exploration_2,
                cache_opponent_moves=args.cache_opponent_moves_2,
                obj_rand=args.obj_rand_2,
                alpha=args.alpha_2,
                pop_size=args.pop_size_2,
                mirror_model=args.mirror_model_2,
                initialize_populations_with_policy = matchup[1]["initialize_populations_with_policy"],
        )
        agent1_wins = 0
        agent2_wins = 0

        for mapstruct, buffer_path in zip(maps, buffer_paths):
            agent1_map_wins = 0
            agent2_map_wins = 0
            replay_buffer = ReplayBuffer()
            for i in range(args.num_games):

                print(f"\n\nStarting game {i+1} out of {args.num_games} on {mapstruct.name}")

                data = {
                    "self-play": False,
                    "map": mapstruct.id,
                    "turns": [],
                    "winner": None
                }

                game = risk.LocalGameManager(mapstruct)

                callbacks = [risk.standard_callback]
                if args.output_dir:
                    callbacks.append(risk.record_data_callback(data))
                if args.surrender_thresh > 0:
                    callbacks.append(risk.early_terminate_callback(args.surrender_thresh))
                if args.tie_after > 0:
                    callbacks.append(risk.tie_callback(args.tie_after))

                result = game.play_loop(
                    bot1,
                    bot2,
                    callback=risk.compose_callbacks(*callbacks),
                )

                data["winner"] = result
                replay_buffer.collect_training_data(data["turns"], mapstruct, 1, 2)

                if result == 1:
                    agent1_wins += 1
                    agent1_map_wins += 1
                elif result == 2:
                    agent2_wins += 1
                    agent2_map_wins += 1
                else:
                    agent1_wins += 0.5
                    agent1_map_wins += 0.5
                    agent2_wins += 0.5
                    agent2_map_wins += 0.5

                print("\n")
                print(f"{matchup[0]['name']} wins: {agent1_map_wins} ({agent1_map_wins/(i+1)*100})%")
                print(f"{matchup[1]['name']} wins: {agent2_map_wins} ({agent2_map_wins/(i+1)*100}%)")
                print("\n")

                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))
                
                print("\n\n")

            replay_buffer.save(buffer_path)

            print("\n\n")
            print(f"Map {mapstruct.name} Results")
            print(f"{matchup[0]['name']} wins: {agent1_map_wins} ({(agent1_map_wins/args.num_games)*100})%")
            print(f"{matchup[1]['name']} wins: {agent2_map_wins} ({(agent2_map_wins/args.num_games)*100}%)")
            print("\n\n")
    
        print("\n\n")
        print(f"Overall Results for matchup between {matchup[0]['name']} and {matchup[1]['name']}")
        print(f"{matchup[0]['name']} wins: {agent1_wins} ({(agent1_wins/args.num_games*len(maps))*100})%")
        print(f"{matchup[1]['name']} wins: {agent2_wins} ({(agent2_wins/args.num_games*len(maps))*100}%)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--surrender-thresh", type=float, default=0.01, help="Terminate early if a player has this probability of winning")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--num-games", type=int, default=50, help="Number of games to play for collecting data")
    parser.add_argument("--map-cache", type=str, default=None, help="Directory to use for map caches")
    parser.add_argument("--max-depth-1", type=int, default=25, help="")
    parser.add_argument("--max-depth-2", type=int, default=25, help="")
    parser.add_argument("--policy-trust-1", type=float, default=1.0, help="")
    parser.add_argument("--policy-trust-2", type=float, default=1.0, help="")
    parser.add_argument("--moves-consider-1", type=int, default=20, help="")
    parser.add_argument("--moves-consider-2", type=int, default=20, help="")
    parser.add_argument("--time-limit-1", type=float, default=10, help="")
    parser.add_argument("--time-limit-2", type=float, default=10, help="")
    parser.add_argument("--exploration-1", type=float, default=0.35, help="")
    parser.add_argument("--exploration-2", type=float, default=0.35, help="")
    parser.add_argument("--cache-opponent-moves-1", type=strtobool, default=False, help="")
    parser.add_argument("--cache-opponent-moves-2", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand-1", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand-2", type=strtobool, default=False, help="")
    parser.add_argument("--alpha-1", type=float, default="inf", help="")
    parser.add_argument("--alpha-2", type=float, default="inf", help="")
    parser.add_argument("--model-type-1", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--model-type-2", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--pop-size-1", type=int, default=50, help="")
    parser.add_argument("--pop-size-2", type=int, default=50, help="")
    parser.add_argument("--mirror-model-1", type=strtobool, default=False, help="")
    parser.add_argument("--mirror-model-2", type=strtobool, default=False, help="")
    parser.add_argument("--rounds-1", type=int, default=1, help="")
    parser.add_argument("--rounds-2", type=int, default=1, help="")
    parser.add_argument("--tie-after", type=int, default=50, help="Number of turns to play before declaring a tie")
    __main__(parser.parse_args())