def compose_callbacks(*cbs):
    def func(*args, **kwargs):
        result = None
        for cb in cbs:
            cb_result = cb(*args, **kwargs)
            result = cb_result if result is None else result
        return result
    return func

def standard_callback(bots, mapstate, turn):
    winrates = {bot: 0.5 * bot.root_node.win_value / bot.root_node.visits + 0.5 for bot in bots}
    times = {bot: bot.elapsed for bot in bots}
    print(f"Turn {turn:2}:")
    for bot in bots:
        print(f"  Winrate:{100*winrates[bot]:6.2f}% ({times[bot]:.2f}s)")


def record_data_callback(data):
    def callback(bots, mapstate, turn):
        data["turns"].append({
            "owner": mapstate.owner.tolist(),
            "armies": mapstate.armies.tolist(),
            "win_value": [int(bot.root_node.win_value) for bot in bots],
            "visits": [int(bot.root_node.visits) for bot in bots],
            "moves": [[child.move.to_json() for child in bot.root_node.children] for bot in bots],
            "move_probs": [[child.visits / bot.root_node.visits for child in bot.root_node.children] for bot in bots],
            "time": [bot.elapsed for bot in bots],
        })
    return callback

def early_terminate_callback(threshold):
    def callback(bots, mapstate, turn):
        early_winner = None
        for bot in bots:
            p = bot.win_prob()
            if p > 1 - threshold:
                if early_winner is not None:
                    return None # two players think they will win
                # this bot thinks it will win
                early_winner = bot.player
            elif p > threshold:
                # this bot thinks the winner is still unknown
                return None
            else:
                # this bot will not challenge the early_winner
                pass
        if early_winner is not None:
            print(f"All players have agreed that Player {early_winner} will win with {100-threshold*100:.2f}% confidence")
        return early_winner
    return callback

def tie_callback(num_turns):
    def callback(bots, mapstate, turn):
        if turn > num_turns:
            p1 = bots[0].win_prob()
            p2 = bots[1].win_prob()
            if 0.49 <= p1 <= 0.51 and 0.49 <= p2 <= 0.51:
                print(f"Game tied after {num_turns} turns")
                return 0
            elif turns > 2*num_turns:
                print(f"Forced tied after {2*num_turns} turns")
                return 0
        return None
    return callback

