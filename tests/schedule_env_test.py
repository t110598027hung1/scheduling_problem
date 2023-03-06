import time
import random
import numpy as np
from env.schedule_env import JSSP


def test_reset_is_correct():
    env = JSSP()
    jss_data = [
        [[30, 40, 10],
         [20, 30, 30],
         [20, 30, 30]],
        [[1, 2, 3],
         [3, 1, 2],
         [3, 2, 1]]
    ]
    num_of_machine = {1: 1, 2: 2, 3: 3}
    for _ in range(2):
        feature, _ = env.reset(jss_data, num_of_machine)
        # basic information
        assert env.duration.tolist() == [[[30, 30, 30], [40, 40, 40], [10, 10, 10]],
                                         [[20, 20, 20], [30, 30, 30], [30, 30, 30]],
                                         [[20, 20, 20], [30, 30, 30], [30, 30, 30]]]
        assert env.machines.tolist() == [[[1, 0, 0], [2, 3, 0], [4, 5, 6]],
                                         [[4, 5, 6], [1, 0, 0], [2, 3, 0]],
                                         [[4, 5, 6], [2, 3, 0], [1, 0, 0]]]
        assert env.machines_map == {1: [1], 2: [2, 3], 3: [4, 5, 6]}
        assert env.machines_last_node == [None, None, None, None, None, None, None]
        assert env.num_of_jobs == 3
        assert env.num_of_tasks == 3
        assert env.num_of_slots == 3
        assert env.num_of_nodes == 27
        assert env.array_shape == (3, 3, 3)
        # adjacency matrix
        expected_adj_index = [(i, i) for i in range(27)]
        expected_adj_index += [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
                               (3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8),
                               (9, 12), (9, 13), (9, 14), (10, 12), (10, 13), (10, 14), (11, 12), (11, 13), (11, 14),
                               (12, 15), (12, 16), (12, 17), (13, 15), (13, 16), (13, 17), (14, 15), (14, 16), (14, 17),
                               (18, 21), (18, 22), (18, 23), (19, 21), (19, 22), (19, 23), (20, 21), (20, 22), (20, 23),
                               (21, 24), (21, 25), (21, 26), (22, 24), (22, 25), (22, 26), (23, 24), (23, 25), (23, 26)]
        assert sorted(list(env.adj_index)) == sorted(expected_adj_index)
        # feature
        assert env.machines_start_time.tolist() == [0, 0, 0, 0, 0, 0, 0]
        assert env.node_end_time.tolist() == [[[30, 30, 30], [70, 70, 70], [80, 80, 80]],
                                              [[20, 20, 20], [50, 50, 50], [80, 80, 80]],
                                              [[20, 20, 20], [50, 50, 50], [80, 80, 80]]]
        assert env.finished_mask.tolist() == [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        # mask
        assert env.feasible_tasks.tolist() == [0, 0, 0]
        assert env.node_index.tolist() == [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                                           [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                                           [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]
        assert env.node_masks.tolist() == [[[False, True, True], [False, False, True], [False, False, False]],
                                           [[False, False, False], [False, True, True], [False, False, True]],
                                           [[False, False, False], [False, False, True], [False, True, True]]]
        # reward
        assert env.makespan == 80
        # solution
        assert len(env.solution_sequence) == 0
        assert len(env.solution_set) == 0


def test_get_feature():
    env = JSSP()
    jss_data = [
        [[30, 40, 10],
         [20, 30, 30],
         [20, 30, 30]],
        [[1, 2, 3],
         [3, 1, 2],
         [3, 2, 1]]
    ]
    num_of_machine = {1: 1, 2: 2, 3: 3}
    feature, _ = env.reset(jss_data, num_of_machine)
    assert np.array_equal(feature, np.array([
        [0.03, 0.00, 0], [0.03, 0.00, 0], [0.03, 0.00, 0],
        [0.07, 0.03, 0], [0.07, 0.03, 0], [0.07, 0.03, 0],
        [0.08, 0.07, 0], [0.08, 0.07, 0], [0.08, 0.07, 0],
        [0.02, 0.00, 0], [0.02, 0.00, 0], [0.02, 0.00, 0],
        [0.05, 0.02, 0], [0.05, 0.02, 0], [0.05, 0.02, 0],
        [0.08, 0.05, 0], [0.08, 0.05, 0], [0.08, 0.05, 0],
        [0.02, 0.00, 0], [0.02, 0.00, 0], [0.02, 0.00, 0],
        [0.05, 0.02, 0], [0.05, 0.02, 0], [0.05, 0.02, 0],
        [0.08, 0.05, 0], [0.08, 0.05, 0], [0.08, 0.05, 0]],
        dtype=np.single))


def test_get_info():
    env = JSSP()
    jss_data = [
        [[30, 40, 10],
         [20, 30, 30],
         [20, 30, 30]],
        [[1, 2, 3],
         [3, 1, 2],
         [3, 2, 1]]
    ]
    num_of_machine = {1: 1, 2: 2, 3: 3}
    _, info = env.reset(jss_data, num_of_machine)
    assert len(info['adj_tuple'][0]) == len(info['adj_tuple'][1]) == len(info['adj_tuple'][2])
    assert info['adj_tuple'][3] == 27
    assert np.array_equal(info['candidate'], np.array([0, 1, 2, 9, 10, 11, 18, 19, 20],
                                                      dtype=np.int64))
    assert np.array_equal(info['node_mask'], np.array([False, True, True, False, False, False, False, False, False],
                                                      dtype=np.int64))


def test_step_and_done():
    env = JSSP()
    jss_data = [
        [[30, 40, 10],
         [20, 30, 30],
         [20, 30, 30]],
        [[1, 2, 3],
         [3, 1, 2],
         [3, 2, 1]]
    ]
    num_of_machine = {1: 1, 2: 2, 3: 3}
    expected_dones = [False, False, False, False, False, False, False, False, True]

    # simulation
    def simulation():
        env.reset(jss_data, num_of_machine)
        for index, action in enumerate(actions):
            _, reward, done, _, _ = env.step(action)
            assert reward == expected_rewards[index]
            assert done == expected_dones[index]
        assert env.makespan == expected_makespan

    actions = [0, 18, 9, 21, 12, 3, 6, 15, 24]
    expected_rewards = [0, 0, -20, 0, 0, 0, 0, -20, 0]
    expected_makespan = 120
    simulation()
    actions = [0, 18, 10, 21, 12, 4, 8, 16, 24]
    expected_rewards = [0, 0, 0, 0, -10, 0, 0, -10, 0]
    expected_makespan = 100
    simulation()
    actions = [0, 18, 10, 21, 12, 4, 8, 15, 24]
    expected_rewards = [0, 0, 0, 0, -10, 0, 0, 0, 0]
    expected_makespan = 90
    simulation()


def test_random_choice_action():
    seed = 200
    episode = 10
    num_of_jobs = 6
    num_of_machines = 6
    num_of_each_machine = 2

    start_time = time.process_time()
    env = JSSP()
    total_makespan = []
    for i in range(episode):
        random.seed(seed + i)
        np.random.seed(seed + i)
        d = [[random.randint(1, 99) for _ in range(num_of_machines)] for _ in range(num_of_jobs)]
        m = [random.sample(range(num_of_machines), num_of_machines) for _ in range(num_of_jobs)]
        num_of_machine = {m: num_of_each_machine for m in range(num_of_machines)}
        _, info = env.reset([d, m], num_of_machine)
        done = False
        while not done:
            probability = np.ones(info['node_mask'].shape)
            probability[info['node_mask']] = 0
            probability = probability / probability.sum()
            action = np.random.choice(info['candidate'], p=probability)
            feature, _, done, _, info = env.step(action)
        total_makespan.append(env.makespan)
    print('\n')
    print('episode:', episode, 'size:', num_of_jobs, 'x', num_of_machines, 'x', num_of_each_machine)
    print('run time:', time.process_time() - start_time, 's')
    print('avg makespan:', sum(total_makespan) / episode)
