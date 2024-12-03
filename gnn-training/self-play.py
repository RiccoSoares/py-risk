import datetime as dt
import argparse
import json
import pickle
import os
from distutils.util import strtobool

import risk
import risk.custom_maps as custom_maps
from risk.replay_buffer import ReplayBuffer
from risk.nn import Model5

try:
    from risk.nn import *
except ImportError:
    pass

def __main__(args):
    mapstruct = custom_maps.create_italy_map()
    network = Model12()

    if (args.model_path is None):
        network = None
    else:
        with open(args.model_path, 'rb') as f:
            network.load_state_dict(pickle.load(f))

        # Set to evaluation mode
        network.eval()

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    if args.replay_buffer_path:
        replay_buffer.load(args.replay_buffer_path)

    for i in range(args.num_games):

        print(f"\n\nStarting game {i+1} out of {args.num_games} on Italy")

        data = {
            "self-play": True,
            "map": 1,
            "turns": [],
            "winner": None
        }


        bot1 = args.model_type(
                None, 1, 2, network,
                iters=args.iter,
                max_depth=args.max_depth,
                trust_policy=args.policy_trust,
                moves_to_consider=args.moves_consider,
                timeout=args.time_limit,
                exploration=args.exploration,
                cache_opponent_moves=args.cache_opponent_moves,
                obj_rand=args.obj_rand,
                alpha=args.alpha,
                pop_size=args.pop_size,
                mirror_model=args.mirror_model,
        )
        bot2 = args.model_type(
                None, 2, 1, network,
                iters=args.iter,
                max_depth=args.max_depth,
                trust_policy=args.policy_trust,
                moves_to_consider=args.moves_consider,
                timeout=args.time_limit,
                exploration=args.exploration,
                cache_opponent_moves=args.cache_opponent_moves,
                obj_rand=args.obj_rand,
                alpha=args.alpha,
                pop_size=args.pop_size,
                mirror_model=args.mirror_model,
        )
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
        replay_buffer.collect_training_data(data["turns"], mapstruct, 1, 2)
        print(f"Game complete: Player {result} Won")
        print("\n")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))
            
        buffer_savepth = args.replay_buffer_path if args.replay_buffer_path else f"replay-buffer/replay_buffer.pkl"
        
        replay_buffer.save(buffer_savepth)
        print(f'Replay buffer saved and contains {len(replay_buffer)} experiences.')
        
        print("\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--surrender-thresh", type=float, default=0.0, help="Terminate early if a player has this probability of winning")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn for player")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play for collecting data")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Capacity of the replay buffer")
    parser.add_argument("--map-cache", type=str, default=None, help="Directory to use for map caches")
    parser.add_argument("--max-depth", type=int, default=25, help="")
    parser.add_argument("--policy-trust", type=float, default=1.0, help="")
    parser.add_argument("--moves-consider", type=int, default=20, help="")
    parser.add_argument("--time-limit", type=float, default=float("inf"), help="")
    parser.add_argument("--exploration", type=float, default=0.35, help="")
    parser.add_argument("--cache-opponent-moves", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand", type=strtobool, default=False, help="")
    parser.add_argument("--alpha", type=float, default="inf", help="")
    parser.add_argument("--model-type", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--pop-size", type=int, default=50, help="")
    parser.add_argument("--mirror-model", type=strtobool, default=False, help="")
    parser.add_argument("--model-path", type=str, help="Path to the initialized model")
    parser.add_argument("--replay-buffer-path", type=str, help="Path to the replay buffer file")
    __main__(parser.parse_args())
