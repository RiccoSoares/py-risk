import datetime as dt
import argparse
import json
import pickle
import os
from distutils.util import strtobool

import risk
import risk.custom_maps as custom_maps
from risk.nn import Model5

try:
    from risk.nn import *
except ImportError:
    pass

def __main__(args):
    latest_gnn = Model15()
    previous_gnn = Model15()

    latest_gnn.load_state_dict(pickle.load('model-weights/it1.pkl'))
    previous_gnn.load_state_dict(pickle.load('model-weights/it0.pkl'))

    # Set to evaluation mode
    latest_gnn.eval()
    previous_gnn.eval()

    latest_agent = args.model_type_1(
                None, 1, 2, latest_gnn,
                iters=150,
                max_depth=args.max_depth_1,
                trust_policy=args.policy_trust_1,
                moves_to_consider=args.moves_consider_1,
                timeout=args.time_limit_1,
                exploration=args.exploration_1,
                cache_opponent_moves=args.cache_opponent_moves_1,
                obj_rand=args.obj_rand_1,
                alpha=args.alpha_1,
                pop_size=args.pop_size_1,
                mirror_model=args.mirror_model_1,
        )

    previous_agent = args.model_type_2(
                None, 2, 1, previous_gnn,
                iters=150,
                max_depth=args.max_depth_2,
                trust_policy=args.policy_trust_2,
                moves_to_consider=args.moves_consider_2,
                timeout=args.time_limit_2,
                exploration=args.exploration_2,
                cache_opponent_moves=args.cache_opponent_moves_2,
                obj_rand=args.obj_rand_2,
                alpha=args.alpha_2,
                pop_size=args.pop_size_2,
                mirror_model=args.mirror_model_2,
        )
    
    baseline = args.model_type_2(
                None, 2, 1, None,
                iters=300,
                max_depth=args.max_depth_2,
                trust_policy=args.policy_trust_2,
                moves_to_consider=args.moves_consider_2,
                timeout=args.time_limit_2,
                exploration=args.exploration_2,
                cache_opponent_moves=args.cache_opponent_moves_2,
                obj_rand=args.obj_rand_2,
                alpha=args.alpha_2,
                pop_size=args.pop_size_2,
                mirror_model=args.mirror_model_2,
        )

    matchups = [(latest_agent, previous_agent, 'Previous Iteration'), (latest_agent, baseline, 'Baseline')]
    maps = [custom_maps.create_simple_map(), custom_maps.create_banana_map(), custom_maps.create_owl_island_map()]

    for matchup in matchups:
        latest_agent_wins = 0
        opp_wins = 0

        for mapstruct in maps:
            for i in range(args.num_games):

                print(f"\n\nStarting game {i+1} out of {args.num_games} on {mapstruct.name}")

                data = {
                    "self-play": False,
                    "map": mapstruct.id,
                    "turns": [],
                    "winner": None
                }


                bot1 = matchup[0]
                bot2 = matchup[1]
                game = risk.LocalGameManager(mapstruct)

                callbacks = [risk.standard_callback]
                if args.output_dir:
                    callbacks.append(risk.record_data_callback(data))
                if args.surrender_thresh > 0:
                    callbacks.append(risk.early_terminate_callback(args.surrender_thresh))

                result = game.play_loop(
                    bot1,
                    bot2,
                    callback=risk.compose_callbacks(*callbacks),
                )

                data["winner"] = result
                if result == 1:
                    latest_agent_wins += 1
                else:
                    opp_wins += 1
                print(f"Game {i} complete: Player {result} Won")
                print(f"Latest Bot wins: {latest_agent_wins} ({latest_agent_wins/(i+1)*100})%")
                print(f"{matchup[2]} Bot wins: {opp_wins} ({opp_wins/(i+1)*100}%)")
                print("\n")

                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))
                
                print("\n\n")
            print(f"Latest Bot wins: {latest_agent_wins} ({latest_agent_wins/args.num_games*100})%")
            print(f"{matchup[2]} wins: {opp_wins} ({opp_wins/args.num_games*100}%)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--surrender-thresh", type=float, default=0.01, help="Terminate early if a player has this probability of winning")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play for collecting data")
    parser.add_argument("--map-cache", type=str, default=None, help="Directory to use for map caches")
    parser.add_argument("--max-depth-1", type=int, default=25, help="")
    parser.add_argument("--max-depth-2", type=int, default=25, help="")
    parser.add_argument("--policy-trust-1", type=float, default=1.0, help="")
    parser.add_argument("--policy-trust-2", type=float, default=1.0, help="")
    parser.add_argument("--moves-consider-1", type=int, default=20, help="")
    parser.add_argument("--moves-consider-2", type=int, default=20, help="")
    parser.add_argument("--time-limit-1", type=float, default=float("inf"), help="")
    parser.add_argument("--time-limit-2", type=float, default=float("inf"), help="")
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
    __main__(parser.parse_args())