import datetime as dt
import argparse
import json
import pickle
import os
import gzip
from distutils.util import strtobool

import risk
import risk.custom_maps as custom_maps
from risk.replay_buffer import ReplayBuffer
from risk.nn import Model15

def __main__(args):
    network = Model15()
    it_num = 2

    with open(f'model-weights/{it_num}.pkl', 'rb') as f:
        network.load_state_dict(pickle.load(f))

    target_experiences = 10000

    italy_map = custom_maps.create_italy_map()
    simple_map = custom_maps.create_simple_map()
    banana_map = custom_maps.create_banana_map()
    owl_island_map = custom_maps.create_owl_island_map()

    italy_replay_buffer = ReplayBuffer(target_experiences + 2000)
    owl_island_replay_buffer = ReplayBuffer(target_experiences + 2000)
    banana_replay_buffer = ReplayBuffer(target_experiences + 2000)
    simple_replay_buffer = ReplayBuffer(target_experiences + 2000)

    italy_buffer_path = f"replay-buffer/{it_num}/italy.pkl"
    owl_island_buffer_path = f"replay-buffer/{it_num}/owl_island.pkl"
    banana_buffer_path = f"replay-buffer/{it_num}/banana.pkl"
    simple_buffer_path = f"replay-buffer/{it_num}/simple.pkl"

    if os.path.exists(italy_buffer_path):
        italy_replay_buffer.load(italy_buffer_path)

    if os.path.exists(owl_island_buffer_path):
        owl_island_replay_buffer.load(owl_island_buffer_path)

    if os.path.exists(banana_buffer_path):
        banana_replay_buffer.load(banana_buffer_path)

    if os.path.exists(simple_buffer_path):
        simple_replay_buffer.load(simple_buffer_path)

    training_setups = [
        (italy_map, italy_replay_buffer, italy_buffer_path),
        (simple_map, simple_replay_buffer, simple_buffer_path),
        (banana_map, banana_replay_buffer, banana_buffer_path),
        (owl_island_map, owl_island_replay_buffer, owl_island_buffer_path),
    ]

    for mapstruct, replay_buffer, buffer_path in training_setups:
        
        game_count = 0
        while len(replay_buffer) < target_experiences:
            print(f"\n\nStarting game {game_count+1} on {mapstruct.name}")

            data = {
                "self-play": True,
                "map": mapstruct.id,
                "turns": [],
                "winner": None
            }


            bot1 = args.model_type(
                    None, 1, 2, network,
                    iters=150,
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
                    iters=150,
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
            if result == 0:
                print("Game complete: Tie")
            else:
                print(f"Game complete: Player {result} Won")
            print("\n")

            os.makedirs('results', exist_ok=True)
            json.dump(data, open(f"results/{dt.datetime.now()}.json", "w"))
            
            replay_buffer.save(buffer_path)
            print(f'Replay buffer saved and contains {len(replay_buffer)} experiences.')
            print("\n\n")
            game_count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--surrender-thresh", type=float, default=0.0, help="Terminate early if a player has this probability of winning")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn for player")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
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
    parser.add_argument("--tie-after", type=int, default=50, help="Number of turns to play before declaring a tie")
    __main__(parser.parse_args())
