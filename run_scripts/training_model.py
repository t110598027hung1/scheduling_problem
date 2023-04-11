import os
import csv
import torch
import random
import datetime
import numpy as np
from algorithm.ppo import PPO, Memory
from env.util import data_generator, graph_pool_weights, load_data_set, get_device
from env.dynamic_env import TimeBasedJSSP
from run_scripts.validation import validation
from torch.distributions.categorical import Categorical


def select_action(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return s, dist.log_prob(s)


def write_log(file_name, row_data):
    with open(file_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)


if __name__ == '__main__':
    num_envs = 4
    num_of_jobs = 9
    num_of_machines = 3
    num_of_tasks = 3
    min_duration = 1
    max_duration = 99
    episode = 15000
    log_frequency = 10
    verification_frequency = 50
    validation_file_name = '9_3_1_99.npy'
    model_folder_name = '../models/' + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '/'

    env = TimeBasedJSSP()
    memories = [Memory() for _ in range(num_envs)]
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
    validation_set = load_data_set('../validation_data/' + validation_file_name)

    for i_update in range(1, episode + 1):
        ep_rewards = [0] * num_envs
        for env_id in range(num_envs):
            feature, info = env.reset(data=data_generator(num_of_jobs=num_of_jobs,
                                                          num_of_machines=num_of_machines,
                                                          num_of_tasks=num_of_tasks,
                                                          min_duration=min_duration,
                                                          max_duration=max_duration),
                                      num_of_jobs=num_of_jobs,
                                      num_of_machine_type=num_of_machines,
                                      random_seed=random.randint(1, 1048576))
            # rollout the env
            while True:
                fea_tensor = torch.from_numpy(feature).to(torch_device)
                pool_tensor = graph_pool_weights('average', env.num_of_nodes, torch_device)
                adj_tensor = torch.sparse_coo_tensor(
                    indices=torch.tensor([info['adj_matrix']['row'], info['adj_matrix']['col']]),
                    values=torch.ones(info['adj_matrix']['value']),
                    size=torch.Size(info['adj_matrix']['size']),
                    dtype=torch.float).to(torch_device)
                cand_tensor = torch.tensor(info['matrix_candidate']).to(torch_device)
                mask_tensor = torch.from_numpy(np.array(info['node_mask'])).to(torch_device)
                with torch.no_grad():
                    pi, _ = ppo.policy_old(x=fea_tensor,
                                           graph_pool=pool_tensor,
                                           adj=adj_tensor,
                                           candidate=cand_tensor.unsqueeze(0),
                                           mask=mask_tensor.unsqueeze(0))
                    a_idx, log_probs = select_action(pi)
                action = info['node_candidate'][a_idx]
                feature, reward, done, _, info = env.step(action)
                ep_rewards[env_id] += reward

                # saving episode data
                memories[env_id].features.append(fea_tensor)
                memories[env_id].graph_pools.append(pool_tensor)
                memories[env_id].adjacencies.append(adj_tensor)
                memories[env_id].candidates.append(cand_tensor)
                memories[env_id].masks.append(mask_tensor)
                memories[env_id].actions_index.append(a_idx)
                memories[env_id].log_probs.append(log_probs)
                memories[env_id].rewards.append(reward)
                memories[env_id].dones.append(done)
                if done:
                    break

        loss, v_loss = ppo.update(memories)
        for memory in memories:
            memory.clear_memory()

        if not os.path.exists(model_folder_name):
            os.makedirs(model_folder_name)
            header = ['Datetime', 'Episode', 'Last_Reward', 'Mean_Loss', 'Mean_V_Loss']
            write_log(model_folder_name + '!training_log.csv', header)
            header = ['Datetime', 'Episode', 'Average_Makespan']
            write_log(model_folder_name + '!validation.csv', header)

        # log results
        if i_update % log_frequency == 0:
            mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
            row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_update, mean_rewards_all_env, loss, v_loss]
            write_log(model_folder_name + '!training_log.csv', row)
            print('{}\t Episode {}\t Last_Reward: {:.2f}\t Mean_Loss: {:.6f}\t Mean_V_Loss: {:.6f}\t'.format(*row))

        # validation
        if i_update % verification_frequency == 0:
            average_makespan = validation(validation_set, ppo.policy_old, 'RL', device=torch_device)
            row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_update, average_makespan]
            write_log(model_folder_name + '!validation.csv', row)
            print('validation average makespan: {:.1f}'.format(average_makespan))

            # save model
            torch.save(ppo.policy.state_dict(),
                       model_folder_name + f'{num_of_jobs}_{num_of_machines}_{i_update}_{int(average_makespan)}.pth')
