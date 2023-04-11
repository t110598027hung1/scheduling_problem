import torch
import torch.nn as nn
from copy import deepcopy
from algorithm.policy.actor_critic import ActorCritic
from torch.distributions.categorical import Categorical


def evaluate_actions(p, a_idx):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(a_idx).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


class Memory:
    def __init__(self):
        self.adjacencies = []
        self.features = []
        self.candidates = []
        self.masks = []
        self.graph_pools = []
        self.actions_index = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def clear_memory(self):
        del self.adjacencies[:]
        del self.features[:]
        del self.candidates[:]
        del self.masks[:]
        del self.graph_pools[:]
        del self.actions_index[:]
        del self.rewards[:]
        del self.dones[:]
        del self.log_probs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.input_dim = input_dim
        self.device = device

        self.policy = ActorCritic(num_layers=num_layers,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device,
                                  aggregate_type='sum')
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.V_loss_2 = nn.MSELoss()
        self.v_loss_coef = 1
        self.p_loss_coef = 2
        self.ent_loss_coef = 0.01

    def update(self, memories):
        device = self.device
        old_rewards = []
        old_log_probs = []
        # store data for all env
        for i in range(len(memories)):
            # monte carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].rewards), reversed(memories[i].dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            # normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            old_rewards.append(rewards)
            # process each env data
            old_log_probs.append(torch.stack(memories[i].log_probs).to(device).squeeze().detach())

        # optimize policy for K epochs
        loss_sum = v_loss_sum = None
        for _ in range(self.k_epochs):
            loss_sum = torch.tensor(0.).to(device)
            v_loss_sum = torch.tensor(0.).to(device)
            for i in range(len(memories)):
                log_probs_list = []
                ent_loss_list = []
                vals_list = []
                for j in range(len(memories[i].adjacencies)):
                    step_pis, step_vals = self.policy(x=memories[i].features[j],
                                                      graph_pool=memories[i].graph_pools[j],
                                                      adj=memories[i].adjacencies[j],
                                                      candidate=memories[i].candidates[j].unsqueeze(0),
                                                      mask=memories[i].masks[j].unsqueeze(0))
                    log_probs2, ent_loss2 = evaluate_actions(step_pis.squeeze(), memories[i].actions_index[j])
                    log_probs_list.append(log_probs2)
                    ent_loss_list.append(ent_loss2)
                    vals_list.append(step_vals.squeeze(0))
                log_probs = torch.stack(log_probs_list).to(device).reshape(-1)
                ent_loss = torch.stack(ent_loss_list).to(device).mean()
                vals = torch.stack(vals_list).to(device)

                ratios = torch.exp(log_probs - old_log_probs[i].detach())
                advantages = old_rewards[i] - vals.view(-1).detach()
                # finding surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), old_rewards[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                # final loss of clipped objective PPO
                loss = self.v_loss_coef * v_loss + self.p_loss_coef * p_loss + self.ent_loss_coef * ent_loss
                loss_sum += loss
                v_loss_sum += v_loss
            # take gradient step
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_sum.mean().item(), v_loss_sum.mean().item()
