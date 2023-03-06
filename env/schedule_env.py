import gym
import numpy as np
from gym import utils
from abc import ABC

END_TIME_NORMALIZE_COEFFICIENT = 1000


class JSSP(gym.Env, utils.EzPickle, ABC):
    def __init__(self):
        utils.EzPickle.__init__(self)
        # basic information
        self.duration: np.ndarray | None = None             # 每個節點需要花費的時間
        self.machines: np.ndarray | None = None             # 每個節點需要使用的機器
        self.machines_map: dict = {}                        # 所有機器的編號 {machine_type: [machine_id]}
        self.machines_last_node: list = []                  # 每台機器前次執行的節點
        self.num_of_jobs: int = 0                           # 工作數量
        self.num_of_tasks: int = 0                          # 任務數量
        self.num_of_slots: int = 0                          # 插槽數量(最大同種類機器數量)
        self.num_of_nodes: int = 0                          # 節點數量
        self.array_shape: tuple | None = None               # 排程問題資料的形狀
        # adjacency matrix
        self.adj_index: set = set()                         # 鄰接稀疏矩陣的座標索引
        # feature
        self.machines_start_time: np.ndarray | None = None  # 每台機器可開始使用的時間
        self.node_end_time: np.ndarray | None = None        # 每個節點最快結束的時間
        self.finished_mask: np.ndarray | None = None        # 每個節點是否完成的標誌(不可選標誌)
        # mask
        self.feasible_tasks: np.ndarray | None = None       # 可選擇的任務
        self.node_index: np.ndarray | None = None           # 節點索引
        self.node_masks: np.ndarray | None = None           # 節點遮罩
        self._range_jobs: list | None = None                # 用於取得節點的工作序列
        # reward
        self.makespan: int = 0                              # 最短總排程時間
        # solution
        self.solution_sequence: list = []                   # 已行動的序列，問題的解
        self.solution_set: set = set()                      # 已行動的集合，用於快速判斷動作是否做過

    def get_feature(self):
        node_start_time = np.maximum(self.node_end_time - self.duration,
                                     self.machines_start_time[self.machines])
        return np.concatenate((
            self.node_end_time.reshape(-1, 1) / END_TIME_NORMALIZE_COEFFICIENT,
            node_start_time.reshape(-1, 1) / END_TIME_NORMALIZE_COEFFICIENT,
            self.finished_mask.reshape(-1, 1)
        ), axis=1)

    def get_info(self):
        adj_index_x, adj_index_y = zip(*self.adj_index)
        adj_value = [1] * len(self.adj_index)
        return {'adj_tuple': (adj_index_y, adj_index_x, adj_value, self.num_of_nodes),
                'candidate': self.node_index[self._range_jobs, self.feasible_tasks].flatten(),
                'node_mask': self.node_masks[self._range_jobs, self.feasible_tasks].flatten()}

    def done(self):
        return True if len(self.solution_sequence) == self.num_of_jobs * self.num_of_tasks else False

    def convert_node_index(self, node_id):
        job_id = node_id // (self.num_of_slots * self.num_of_tasks)
        task_id = node_id // self.num_of_slots % self.num_of_tasks
        slot_id = node_id % self.num_of_slots
        return job_id, task_id, slot_id

    def convert_data(self, data: list):
        self.num_of_jobs = len(data[0])
        self.num_of_tasks = len(data[0][0])
        self.num_of_nodes = self.num_of_jobs * self.num_of_tasks * self.num_of_slots
        self.array_shape = (self.num_of_jobs, self.num_of_tasks, self.num_of_slots)
        self._range_jobs = list(range(self.num_of_jobs))
        self.duration = np.zeros(self.array_shape, dtype=np.single)
        self.machines = np.zeros(self.array_shape, dtype=np.int64)
        for job_id, (job_time, job_machines) in enumerate(zip(data[0], data[1])):
            for task_id, (task_time, task_machine) in enumerate(zip(job_time, job_machines)):
                self.duration[job_id, task_id] = task_time
                for slot_id, machine_id in enumerate(self.machines_map[task_machine]):
                    self.machines[job_id, task_id, slot_id] = machine_id

    def build_adjacency_matrix(self):
        self.adj_index.update((i, i) for i in range(self.num_of_nodes))
        node_id = 0
        slots = self.num_of_slots
        for _ in range(self.num_of_jobs):
            node_id += slots
            for t in range(self.num_of_tasks - 1):
                self.adj_index.update(
                    (current_node_id, next_node_id)
                    for next_node_id in range(node_id, node_id + slots)
                    for current_node_id in range(node_id - slots, node_id))
                node_id += slots

    def build_jobs(self):
        # node_end_time
        self.node_end_time = np.cumsum(self.duration, axis=1)
        # finished_mask
        self.finished_mask = np.zeros(self.array_shape, dtype=np.single)
        # feasible_tasks
        self.feasible_tasks = np.zeros(self.num_of_jobs, dtype=np.int64)
        # node_index
        self.node_index = np.arange(self.num_of_nodes, dtype=np.int64).reshape(self.array_shape)
        # node_masks
        self.node_masks = np.zeros(self.array_shape, dtype=bool)
        self.node_masks[self.machines == 0] = True
        # adj_index
        self.build_adjacency_matrix()
        # makespan
        self.makespan = np.max(self.node_end_time[:, -1])

    def build_machines(self, num_of_machines):
        machine_id = 0
        for machine_type, machine_size in num_of_machines.items():
            self.machines_map[machine_type] = [machine_id := machine_id + 1 for _ in range(machine_size)]
            if machine_size > self.num_of_slots:
                self.num_of_slots = machine_size
        self.machines_last_node = [None] * (machine_id + 1)
        self.machines_start_time = np.zeros(machine_id + 1, dtype=np.single)

    def step(self, node_action: int):
        node_action_3d = self.convert_node_index(node_action)
        job_action = node_action_3d[0]
        task_action = node_action_3d[1]
        if node_action in self.solution_set:
            raise 'This action has already been executed.'

        # update solution
        self.solution_sequence.append(node_action)
        self.solution_set.add(node_action)

        # update adjacency matrix  TODO: delete node connection
        machine_type = self.machines[node_action_3d]
        last_node = self.machines_last_node[machine_type]
        self.machines_last_node[machine_type] = node_action
        if last_node is not None:
            self.adj_index.add((last_node, node_action))

        # update feature
        reward = 0
        machine_start_time = self.machines_start_time[machine_type]
        node_start_time = self.node_end_time[node_action_3d] - self.duration[node_action_3d]
        if node_start_time >= machine_start_time:
            self.machines_start_time[machine_type] = self.node_end_time[node_action_3d]
        else:
            increment = machine_start_time - node_start_time
            self.machines_start_time[machine_type] += self.duration[node_action_3d]
            self.node_end_time[job_action, task_action:] += increment
            job_end_time = np.max(self.node_end_time[job_action, -1])
            reward = self.makespan - job_end_time
            if reward < 0:
                self.makespan = job_end_time
            else:
                reward = 0
        self.finished_mask[node_action_3d] = 1

        # update mask
        if self.num_of_tasks - self.feasible_tasks[job_action] > 1:
            self.feasible_tasks[job_action] += 1
        else:
            self.node_masks[job_action, -1] = True

        return self.get_feature(), reward, self.done(), False, self.get_info()

    def reset(self, data: list | np.ndarray = None, num_of_machines: dict = None):
        # reset basic information
        self.duration = None
        self.machines = None
        self.machines_map.clear()
        self.machines_last_node.clear()
        self.num_of_jobs = 0
        self.num_of_tasks = 0
        self.num_of_slots = 0
        self.num_of_nodes = 0
        self.array_shape = None
        # reset adjacency matrix
        self.adj_index.clear()
        # reset feature
        self.machines_start_time = None
        self.node_end_time = None
        self.finished_mask = None
        # reset mask
        self.feasible_tasks = None
        self.node_index = None
        self.node_masks = None
        self._range_jobs = None
        # reset reward
        self.makespan = 0
        # reset solution
        self.solution_sequence.clear()
        self.solution_set.clear()

        # set jobs and machines
        if data is not None:
            if num_of_machines is None:
                num_of_machines = {m: 1 for machines in data[1] for m in machines}
            self.build_machines(num_of_machines)
            self.convert_data(data)
            self.build_jobs()

        return self.get_feature(), self.get_info()
