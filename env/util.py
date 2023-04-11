import torch
import random
import numpy as np


def graph_pool_weights(graph_pool_type, num_of_nodes, device):
    """
    :param graph_pool_type: optional 'average'
    :param num_of_nodes: number of graph nodes
    :param device: torch device
    :return: torch tensor
    """
    if graph_pool_type == 'average':
        return torch.full(size=(1, num_of_nodes),
                          fill_value=1 / num_of_nodes,
                          dtype=torch.float32,
                          device=device)


def data_generator(num_of_jobs, num_of_machines, num_of_tasks, min_duration, max_duration):
    d = [[random.randint(min_duration, max_duration) for _ in range(num_of_tasks)] for _ in range(num_of_jobs)]
    if num_of_tasks <= num_of_machines:
        m = [random.sample(range(num_of_machines), num_of_tasks) for _ in range(num_of_jobs)]
    else:
        m = [[random.randint(1, num_of_machines) for _ in range(num_of_tasks)] for _ in range(num_of_jobs)]
    return [d, m]


def machine_generator(num_of_machines, min_machine, max_machine):
    num_of_each_machine = {m: random.randrange(min_machine, max_machine) for m in range(num_of_machines)}
    return num_of_each_machine


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    return device


def save_data_set(file_name, size=1, **kwargs):
    np.save(file_name, [{
        'data': data_generator(**kwargs),
        'jobs': kwargs['num_of_jobs'],
        'machines': kwargs['num_of_machines'],
        'random_number': random.randrange(1048576)  # 2^20
    } for _ in range(size)])


def load_data_set(file_name):
    return np.load(file_name, allow_pickle=True)


if __name__ == '__main__':
    random.seed(200)
    n_of_jobs = 150
    n_of_machines = 15
    min_dur = 1
    max_dur = 99
    batch_size = 30
    file_path = f'../validation_data/{n_of_jobs}_{n_of_machines}_{min_dur}_{max_dur}.npy'
    save_data_set(file_name=file_path,
                  size=batch_size,
                  num_of_jobs=n_of_jobs,
                  num_of_machines=n_of_machines,
                  num_of_tasks=n_of_machines,
                  min_duration=min_dur,
                  max_duration=max_dur)
