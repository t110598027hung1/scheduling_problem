import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.policy.mlp import MLPActor
from algorithm.policy.mlp import MLPCritic
from algorithm.policy.graph_cnn import GraphCNN


class ActorCritic(nn.Module):
    def __init__(self,
                 # feature extraction net unique attributes:
                 num_layers,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 aggregate_type,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()

        self.feature_extract = GraphCNN(num_layers=num_layers-1,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        aggregate_type=aggregate_type).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim * 2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                adj,
                candidate,
                mask
                ):
        h_pooled, h_nodes = self.feature_extract(x_concat=x,
                                                 graph_pool=graph_pool,
                                                 adj_block=adj)
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, candidate.size(-1), h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        # concatenate feature
        concat_fea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concat_fea)
        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v
