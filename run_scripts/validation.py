import torch
import random
import numpy as np
from algorithm.ppo import PPO
from env.util import graph_pool_weights, load_data_set, get_device
from env.dynamic_env import TimeBasedJSSP


def validation(validate_set, model, select_action_policy='RL', device='cpu'):
    """
    solve dynamic JSSP.
    :param validate_set: JSSP data (in validation_data folder)
    :param model: set RL policy network if select_action_policy is 'RL'
    :param select_action_policy: 'RL', 'random', 'SPT', 'MWKR', 'MOPNR'
    :param device: pytorch device
    """
    env = TimeBasedJSSP()
    total_makespan = []
    for validate_data in validate_set:
        feature, info = env.reset(data=validate_data['data'],
                                  num_of_jobs=validate_data['jobs'],
                                  num_of_machine_type=validate_data['machines'],
                                  random_seed=validate_data['random_number'])
        while True:
            if select_action_policy == 'RL':
                # reinforcement learning
                fea_tensor = torch.from_numpy(feature).to(device)
                adj_tensor = torch.sparse_coo_tensor(
                    indices=torch.tensor([info['adj_matrix']['row'], info['adj_matrix']['col']]),
                    values=torch.ones(info['adj_matrix']['value']),
                    size=torch.Size(info['adj_matrix']['size']),
                    dtype=torch.float).to(device)
                cand_tensor = torch.tensor(info['matrix_candidate']).to(device)
                mask_tensor = torch.from_numpy(np.array(info['node_mask'])).to(device)
                with torch.no_grad():
                    pi, _ = model(x=fea_tensor,
                                  graph_pool=graph_pool_weights('average', env.num_of_nodes, device),
                                  adj=adj_tensor,
                                  candidate=cand_tensor.unsqueeze(0),
                                  mask=mask_tensor.unsqueeze(0))
                    _, a_idx = pi.squeeze().max(0)
                action = info['node_candidate'][a_idx]

            elif select_action_policy == 'random':
                # random choice
                candidate = [node for node, mask in zip(info['node_candidate'], info['node_mask']) if not mask]
                action = random.choice(candidate)

            elif select_action_policy == 'SPT':
                # shortest processing time
                candidate = [node for node, mask in zip(info['node_candidate'], info['node_mask']) if not mask]
                processing_time, machine_start_time = [], []
                for node_id in candidate:
                    _, _, machine, duration = env.nodes_map[node_id]
                    processing_time.append(duration)
                    machine_start_time.append(env.machines_start_time[machine])
                priority_queue = sorted(zip(candidate, processing_time, machine_start_time),
                                        key=lambda x: (x[1], x[2]))
                bast = priority_queue[0][1:]
                candidate_action = [tup[0] for tup in priority_queue if tup[1:] == bast]
                action = random.choice(candidate_action)

            elif select_action_policy == 'MWKR':
                # most work remaining
                candidate = [node for node, mask in zip(info['node_candidate'], info['node_mask']) if not mask]
                work_remaining_time, machine_start_time = [], []
                for node_id in candidate:
                    job, task, machine, _ = env.nodes_map[node_id]
                    work_remaining_time.append(-sum(
                        env.nodes_map[post_task[0]][3] for post_task in env.jobs_map[job][task:]))
                    machine_start_time.append(env.machines_start_time[machine])
                priority_queue = sorted(zip(candidate, work_remaining_time, machine_start_time),
                                        key=lambda x: (x[1], x[2]))
                bast = priority_queue[0][1:]
                candidate_action = [tup[0] for tup in priority_queue if tup[1:] == bast]
                action = random.choice(candidate_action)

            elif select_action_policy == 'MOPNR':
                # most operation remaining
                candidate = [node for node, mask in zip(info['node_candidate'], info['node_mask']) if not mask]
                operation_remaining_time, machine_start_time = [], []
                for node_id in candidate:
                    job, task, machine, _ = env.nodes_map[node_id]
                    operation_remaining_time.append(-len(env.jobs_map[job][task:]))
                    machine_start_time.append(env.machines_start_time[machine])
                priority_queue = sorted(zip(candidate, operation_remaining_time, machine_start_time),
                                        key=lambda x: (x[1], x[2]))
                bast = priority_queue[0][1:]
                candidate_action = [tup[0] for tup in priority_queue if tup[1:] == bast]
                action = random.choice(candidate_action)

            else:
                raise ValueError('select action policy is not found')

            feature, _, done, _, info = env.step(action)
            if done:
                total_makespan.append(int(env.makespan))
                break
    return sum(total_makespan) / len(validate_set)


if __name__ == '__main__':
    model_file_name = '2023-04-12-004529/9_3_50_419.pth'
    tests_file_name = '9_3_1_99.npy'
    algorithms = ['RL', 'random', 'SPT', 'MWKR', 'MOPNR']

    random.seed(200)
    np.random.seed(200)
    torch.manual_seed(600)
    torch_device = get_device()
    print('Device set to:', torch_device)
    ppo = PPO(lr=2e-5,
              gamma=1,
              k_epochs=1,
              eps_clip=0.2,
              num_layers=3,
              input_dim=3,
              hidden_dim=64,
              num_mlp_layers_feature_extract=2,
              num_mlp_layers_actor=2,
              hidden_dim_actor=32,
              num_mlp_layers_critic=2,
              hidden_dim_critic=32,
              device=torch_device)
    ppo.policy.load_state_dict(torch.load('../models/' + model_file_name, map_location=torch_device))
    validation_set = load_data_set('../validation_data/' + tests_file_name)
    for algorithm in algorithms:
        average_makespan = validation(validation_set, ppo.policy, select_action_policy=algorithm, device=torch_device)
        print('{:>6s}'.format(algorithm), 'average makespan: {:.1f}'.format(average_makespan))
