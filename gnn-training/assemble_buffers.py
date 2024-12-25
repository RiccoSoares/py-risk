import pickle
from risk.replay_buffer import ReplayBuffer


def assemble_buffers(src_buffer, dst_buffer):
    for exp in src_buffer.buffer:
        dst_buffer.add(exp)

    return

def main():
    owl_exps = ReplayBuffer(50000)
    owl_exps.load('replay-buffer/data/owl_island.pkl')
    print('Loaded owl_island replay buffer of size:', len(owl_exps.buffer))

    simple_exps = ReplayBuffer(50000)
    simple_exps.load('replay-buffer/data/simple.pkl')
    print('Loaded simple replay buffer of size:', len(simple_exps.buffer))

    banana_exps = ReplayBuffer(50000)
    banana_exps.load('replay-buffer/data/banana.pkl')
    print('Loaded banana replay buffer of size:', len(banana_exps.buffer))

    owl_it2_exps = ReplayBuffer(50000)
    owl_it2_exps.load('replay-buffer/it2/owl_island.pkl')
    print('Loaded IT2 owl_island replay buffer of size:', len(owl_it2_exps.buffer))

    simple_it2_exps = ReplayBuffer(50000)
    simple_it2_exps.load('replay-buffer/it2/simple.pkl')
    print('Loaded IT2 simple replay buffer of size:', len(simple_it2_exps.buffer))

    banana_it2_exps = ReplayBuffer(50000)
    banana_it2_exps.load('replay-buffer/it2/banana.pkl')
    print('Loaded IT2 banana replay buffer of size:', len(banana_it2_exps.buffer))

    assemble_buffers(owl_it2_exps, owl_exps)
    print('Assembled owl_island replay buffer of size:', len(owl_exps.buffer))
    owl_exps.save('replay-buffer/data/owl_island.pkl')

    assemble_buffers(simple_it2_exps, simple_exps)
    print('Assembled simple replay buffer of size:', len(simple_exps.buffer))
    simple_exps.save('replay-buffer/data/simple.pkl')

    assemble_buffers(banana_it2_exps, banana_exps)
    print('Assembled banana replay buffer of size:', len(banana_exps.buffer))
    banana_exps.save('replay-buffer/data/banana.pkl')


if __name__ == "__main__":
    main()