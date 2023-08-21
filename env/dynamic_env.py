import gym
import scipy
import numpy as np
from gym import utils
from abc import ABC
from collections import OrderedDict


class MatrixCSR:
    def __init__(self):
        self.indices: int = 0
        self.node_map: OrderedDict = OrderedDict()
        self.crow_indices: list = [0]
        self.col_indices: list = []

    def get_info(self):
        return {'crow': self.crow_indices,
                'col': self.col_indices,
                'value': len(self.col_indices)}

    def new_full_connect(self, start_layer, new_end_layer):
        """
        創建新的多層全連接節點
        """
        connect_size = len(start_layer) + 1
        transform_start_layer = [self.node_map[i] for i in start_layer]
        for end_node in new_end_layer:
            self.node_map[end_node] = self.indices
            self.crow_indices.append(self.crow_indices[-1] + connect_size)
            self.col_indices += transform_start_layer
            self.col_indices.append(self.indices)
            self.indices += 1

    def add_connect(self, start_node, end_node):
        """
        添加一條現有節點之間的連線
        該方法效能差，請避免大量調用
        """
        start_node = self.node_map[start_node]
        end_node = self.node_map[end_node]
        # update crow indices
        for pos in range(end_node + 1, len(self.crow_indices)):
            self.crow_indices[pos] += 1
        # update col indices
        s_pos = self.crow_indices[end_node]
        t_pos = self.crow_indices[end_node + 1]
        self.col_indices.insert(s_pos, start_node)
        self.col_indices[s_pos:t_pos] = sorted(self.col_indices[s_pos:t_pos])

    def batch_add_connect(self, batch_start_layer, batch_end_layer):
        """
        批量處理: 對已有的節點進行全連接
        """
        connect_buffer = {}
        for start_layer, end_layer in zip(batch_start_layer, batch_end_layer):
            transform_start_layer = [self.node_map[i] for i in start_layer]
            transform_end_layer = [self.node_map[i] for i in end_layer]
            for end_node in transform_end_layer:
                if end_node in connect_buffer:
                    raise ValueError('batch_end_layer has duplicate numbers')
                connect_buffer[end_node] = transform_start_layer
        connect_order = sorted(connect_buffer.keys())
        # update crow indices
        cum_size = 0
        for i, j in zip(connect_order, connect_order[1:] + [self.indices]):
            cum_size += len(connect_buffer[i])
            for pos in range(i, j):
                self.crow_indices[pos + 1] += cum_size
        # update col indices
        for end_node in connect_order:
            s_pos = self.crow_indices[end_node]
            t_pos = self.crow_indices[end_node + 1]
            self.col_indices[s_pos:s_pos] = connect_buffer[end_node]
            self.col_indices[s_pos:t_pos] = sorted(self.col_indices[s_pos:t_pos])

    def delete_node(self, nodes: set):
        """
        刪除節點和其連線
        """
        number = 0
        transform = {}
        nodes_index = {self.node_map[i] for i in nodes}
        for node_id in self.node_map.keys():
            if node_id in nodes:
                number += 1
            else:
                old_index = self.node_map[node_id]
                new_index = old_index - number
                self.node_map[node_id] = new_index
                transform[old_index] = new_index
        new_crow_indices = [0]
        new_col_indices = []
        for index in range(self.indices):
            if index not in nodes_index:
                s_pos = self.crow_indices[index]
                t_pos = self.crow_indices[index + 1]
                new_col_indices += [transform[i] for i in self.col_indices[s_pos:t_pos] if i not in nodes_index]
                new_crow_indices.append(len(new_col_indices))
        # update
        self.indices -= number
        for node_id in nodes:
            self.node_map.pop(node_id)
        self.crow_indices = new_crow_indices
        self.col_indices = new_col_indices

    def clear(self):
        self.indices = 0
        self.node_map.clear()
        self.crow_indices.clear()
        self.col_indices.clear()
        self.crow_indices.append(0)


class MatrixCOO:
    def __init__(self):
        self.indices: int = 0
        self.node_map: OrderedDict = OrderedDict()
        self.row_indices: list = []  # y
        self.col_indices: list = []  # x

    def get_info(self):
        return {'row': self.row_indices,
                'col': self.col_indices,
                'value': len(self.col_indices),
                'size': (self.indices, self.indices)}

    def new_full_connect(self, start_layer, new_end_layer):
        """
        創建新的多層全連接節點
        """
        connect_size = len(start_layer) + 1
        transform_start_layer = [self.node_map[i] for i in start_layer]
        for end_node in new_end_layer:
            self.node_map[end_node] = self.indices
            self.row_indices += [self.node_map[end_node]] * connect_size
            self.col_indices += transform_start_layer
            self.col_indices.append(self.indices)
            self.indices += 1

    def add_connect(self, start_node, end_node):
        """
        添加一條連線
        """
        start_node = self.node_map[start_node]
        end_node = self.node_map[end_node]
        # update indices
        self.col_indices.append(start_node)
        self.row_indices.append(end_node)

    def batch_add_connect(self, batch_start_layer, batch_end_layer):
        """
        批量處理: 對已有的節點進行全連接
        """
        for start_layer, end_layer in zip(batch_start_layer, batch_end_layer):
            transform_start_layer = [self.node_map[i] for i in start_layer]
            transform_end_layer = [self.node_map[i] for i in end_layer]
            for end_node in transform_end_layer:
                self.row_indices += [end_node] * len(start_layer)
                self.col_indices += transform_start_layer

    def delete_node(self, nodes: set):
        """
        刪除節點和其連線
        """
        number = 0
        transform = {}
        nodes_index = {self.node_map[i] for i in nodes}
        for node_id in self.node_map.keys():
            if node_id in nodes:
                number += 1
            else:
                old_index = self.node_map[node_id]
                new_index = old_index - number
                self.node_map[node_id] = new_index
                transform[old_index] = new_index
        new_row_indices = []
        new_col_indices = []
        for row, col in zip(self.row_indices, self.col_indices):
            if not (row in nodes_index or col in nodes_index):
                new_row_indices.append(transform[row])
                new_col_indices.append(transform[col])
        # update
        self.indices -= number
        for node_id in nodes:
            self.node_map.pop(node_id)
        self.row_indices = new_row_indices
        self.col_indices = new_col_indices

    def clear(self):
        self.indices = 0
        self.node_map.clear()
        self.row_indices.clear()
        self.col_indices.clear()


def inject_matrix(token: str):
    """
    :param token: 'csr' or 'coo'
    :return: Instance of Matrix class
    """
    class_name = 'Matrix' + token.upper()
    if class_name in globals():
        return eval(f'{class_name}()')
    raise ValueError(f'{class_name} does not exist')


class DynamicJSSP(gym.Env, utils.EzPickle, ABC):
    # feature named index
    END_TIME = 0
    START_TIME = 1
    FINISHED_MASK = 2

    def __init__(self, matrix_type='coo'):
        utils.EzPickle.__init__(self)
        # basic information
        self.num_of_nodes: int = 0
        self.jobs_map: dict = {}
        self.nodes_map: dict = {}
        self.machines_map: dict = {}
        self.machines_to_type: dict = {}
        self.machines_last_node: dict = {}
        self.machines_start_time: dict = {}
        self._job_id: int = 0
        self._node_id: int = 0
        self._machine_id: int = 0
        # adjacency matrix
        self.machine_tasks_map: dict = {}
        self.matrix = inject_matrix(matrix_type)
        # feature
        self.feature_map: OrderedDict = OrderedDict()
        self.normalize_coefficient = np.array([[1000, 1000, 1]], dtype=np.single)
        # mask
        self.complete_task: dict = {}
        self.candidate_nodes: OrderedDict = OrderedDict()
        self.pool_mask = set()
        # reward
        self.makespan: int = 0
        # solution
        self.machines_info: dict = {}
        self.jobs_info: dict = {}
        self.action_info: list = []

    def get_feature(self):
        """
        獲取節點特徵
        """
        return np.stack(list(self.feature_map.values())) / self.normalize_coefficient

    def get_info(self):
        """
        獲取與圖相關的資訊
        """
        node_mask = []
        node_candidate = []
        for k, v in self.candidate_nodes.items():
            node_mask += [k not in self.complete_task] * len(v)
            node_candidate += v
        return {'adj_matrix': self.matrix.get_info(),
                'node_mask': node_mask,
                'node_candidate': node_candidate,
                'matrix_candidate': [self.matrix.node_map[n] for n in node_candidate]}

    def get_pool_mask(self):
        node_id_mask = {}
        for job_id, job in self.jobs_map.items():
            complete_len = len(job) - self.complete_task.get(job_id, 0)
            for task in job[:complete_len]:
                for node_id in task:
                    if node_id not in self.pool_mask:
                        node_id_mask[node_id] = True
        mask = [False] * self.matrix.indices
        for n, b in node_id_mask.items():
            index = self.matrix.node_map[n]
            mask[index] = b
        return mask

    def done(self):
        """
        判斷模擬是否完成
        """
        return not self.complete_task

    def add_machines(self, machine_type, number, start_time=0):
        """
        添加新機器
        """
        machine_list = self.machines_map.setdefault(machine_type, [])
        if not machine_list:
            self.machine_tasks_map[machine_type] = []
        added_machines_id = list(range(self._machine_id, self._machine_id + number))
        for new_id in added_machines_id:
            self.machines_to_type[new_id] = machine_type
            self.machines_start_time[new_id] = start_time
            self.machines_last_node[new_id] = None
            machine_list.append(new_id)
        self._machine_id += number
        if self.machine_tasks_map[machine_type]:
            self.extend_machine_adjacency_matrix(machine_type, machine_list[-number:])
        self.record_machine_info(machine_type, added_machines_id, 'start', start_time)

    def delete_machines(self, machine_type, delete_index, delete_time=0):
        """
        刪除現有機器
        """
        machine_list = self.machines_map[machine_type]
        if len(delete_index) >= len(machine_list):
            raise ValueError('after delete machines, the number of machines will be less than 1')
        delete_index = sorted(delete_index, reverse=True)
        deleted_machines_id = []
        delete_nodes = set()
        for index in delete_index:
            machine_id = machine_list.pop(index)
            deleted_machines_id.append(machine_id)
            self.machines_to_type.pop(machine_id)
            self.machines_start_time.pop(machine_id)
            self.machines_last_node.pop(machine_id)
            delete_nodes.update(task['current'].pop(index) for task in self.machine_tasks_map[machine_type])
        self.reduced_machine_adjacency_matrix(delete_nodes)
        self.record_machine_info(machine_type, deleted_machines_id, 'end', delete_time)

    def add_jobs(self, duration, machine):
        """
        添加新工作
        """
        job = []
        cum_dur = 0
        for task_indices, (dur, machine_type) in enumerate(zip(duration, machine)):
            cum_dur += dur
            machine_list = self.machines_map[machine_type]
            task = self.create_node(self._job_id, task_indices, machine_list, cum_dur, dur)
            job.append(task)
        # build dependencies between machines and tasks
        job.append([])
        for task_indices, (dur, machine_type) in enumerate(zip(duration, machine)):
            self.machine_tasks_map[machine_type].append({
                'job_id': self._job_id,
                'task_indices': task_indices,
                'duration': dur,
                'pre': job[task_indices - 1],
                'current': job[task_indices],
                'post': job[task_indices + 1]
            })
        job.pop()
        # build tasks conjunction
        self.jobs_map[self._job_id] = job
        self.complete_task[self._job_id] = len(job)
        self.candidate_nodes[self._job_id] = job[0]
        self.makespan = max(self.makespan, cum_dur)
        self._job_id += 1
        self.extend_job_adjacency_matrix(job)

    def create_node(self, job_id, task_indices, machine_list, end_time, duration):
        """
        建立節點特徵
        """
        number = len(machine_list)
        node_id_list = list(range(self._node_id, self._node_id + number))
        self._node_id += number
        self.num_of_nodes += number
        for node_id, machine_id in zip(node_id_list, machine_list):
            self.nodes_map[node_id] = (job_id, task_indices, machine_id, duration)
            self.feature_map[node_id] = np.array([
                # feature 1. expected task the fastest end time
                end_time,
                # feature 2. actual task start time
                max(end_time - duration, self.machines_start_time[machine_id]),
                # feature 3. task completed flag
                0
            ], dtype=np.single)
        return node_id_list

    def remove_node(self, node_set):
        """
        刪除節點
        """
        for node_id in node_set:
            self.nodes_map.pop(node_id)
            self.feature_map.pop(node_id)
        self.num_of_nodes -= len(node_set)

    def extend_machine_adjacency_matrix(self, machine_type, new_machine_list):
        """
        添加機器時擴展鄰接矩陣，並根據機器增加數量擴展任務節點
        """
        start_layers = []
        end_layers = []
        for task in self.machine_tasks_map[machine_type]:
            end_time = self.feature_map[task['current'][0]][DynamicJSSP.END_TIME]
            extend_task = self.create_node(
                task['job_id'], task['task_indices'], new_machine_list, end_time, task['duration'])
            task['current'] += extend_task
            self.matrix.new_full_connect(task['pre'], extend_task)
            start_layers.append(extend_task)
            end_layers.append(task['post'])
        self.matrix.batch_add_connect(start_layers, end_layers)

    def reduced_machine_adjacency_matrix(self, delete_nodes):
        """
        刪除機器時縮減鄰接矩陣
        """
        self.remove_node(delete_nodes)
        self.matrix.delete_node(delete_nodes)

    def extend_job_adjacency_matrix(self, job):
        """
        添加工作時擴展鄰接矩陣
        """
        last_task = []
        for current_task in job:
            self.matrix.new_full_connect(last_task, current_task)
            last_task = current_task

    def record_machine_info(self, machine_type, machine_list, key, value):
        """
        紀錄機器的訊息
        """
        machines_dict = self.machines_info.setdefault(machine_type, {})
        for machine_id in machine_list:
            info_dict = machines_dict.setdefault(machine_id, {})
            info_dict[key] = int(value)

    def record_job_info(self, job_id, machine_id, start_time, end_time):
        """
        紀錄工作排程的訊息
        """
        tasks_list = self.jobs_info.setdefault(job_id, [])
        tasks_list.append({
            'machine_type': self.machines_to_type[machine_id],
            'machine_id': machine_id,
            'start': int(start_time),
            'end': int(end_time),
        })

    def record_action_info(self, job_id, task_indices, machine_id, start_time, end_time):
        """
        紀錄執行動作序列的訊息
        """
        self.action_info.append({
            'job_id': job_id,
            'task_indices': task_indices,
            'machine_type': self.machines_to_type[machine_id],
            'machine_id': machine_id,
            'start': int(start_time),
            'end': int(end_time)
        })

    def step(self, action: int):
        """
        執行動作
        """
        # update adjacency matrix
        job_id, task_indices, machine_id, duration = self.nodes_map[action]
        last_node = self.machines_last_node[machine_id]
        self.machines_last_node[machine_id] = action
        if last_node is not None:
            self.matrix.add_connect(last_node, action)

        # update feature
        # the task must wait for the previous task and the machine to be operated to complete before it can start
        node_end_time = self.feature_map[action][DynamicJSSP.END_TIME]
        increment = max(self.machines_start_time[machine_id] - node_end_time + duration, 0)
        end_time = node_end_time + increment
        self.machines_start_time[machine_id] = end_time
        # feature 1. end time
        for post_task in self.jobs_map[job_id][task_indices:]:
            for node in post_task:
                self.feature_map[node][DynamicJSSP.END_TIME] += increment
        end_node = self.jobs_map[job_id][-1][0]
        max_end_time = self.feature_map[end_node][DynamicJSSP.END_TIME]
        reward = min(self.makespan - max_end_time, 0.)
        self.makespan = max(self.makespan, max_end_time)
        # feature 2. start time
        for k, (j, t, m, dur) in self.nodes_map.items():
            if m == machine_id or (j == job_id and t >= task_indices):
                self.feature_map[k][DynamicJSSP.START_TIME] = max(self.feature_map[k][DynamicJSSP.END_TIME] - dur,
                                                                  self.machines_start_time[m])
        # feature 3. finished mask
        self.feature_map[action][DynamicJSSP.FINISHED_MASK] = 1

        # update mask
        self.complete_task[job_id] -= 1
        if self.complete_task[job_id] == 0:
            self.complete_task.pop(job_id)
        else:
            self.candidate_nodes[job_id] = self.jobs_map[job_id][task_indices + 1]
        self.pool_mask.add(action)

        # update solution
        self.record_job_info(job_id, machine_id, end_time - duration, end_time)
        self.record_action_info(job_id, task_indices, machine_id, end_time - duration, end_time)

        return self.get_feature(), reward, self.done(), False, self.get_info()

    def reset(self, data: list | np.ndarray = None, num_of_machines: dict = None):
        """
        重設環境
        """
        # reset basic information
        self.num_of_nodes = 0
        self.jobs_map.clear()
        self.nodes_map.clear()
        self.machines_map.clear()
        self.machines_to_type.clear()
        self.machines_last_node.clear()
        self.machines_start_time.clear()
        self._job_id = 0
        self._node_id = 0
        self._machine_id = 0
        # reset adjacency matrix
        self.machine_tasks_map.clear()
        self.matrix.clear()
        # reset feature
        self.feature_map.clear()
        # reset mask
        self.complete_task.clear()
        self.candidate_nodes.clear()
        self.pool_mask.clear()
        # reset reward
        self.makespan = 0
        # reset solution
        self.machines_info.clear()
        self.jobs_info.clear()
        self.action_info.clear()

        # set machines and jobs
        if data is not None:
            if num_of_machines is None:
                num_of_machines = {m: 1 for machines in data[1] for m in machines}
            for machine_type, number in num_of_machines.items():
                self.add_machines(machine_type, number)
            for duration, machines in zip(data[0], data[1]):
                self.add_jobs(duration, machines)

            return self.get_feature(), self.get_info()


class TimeBasedJSSP(DynamicJSSP, ABC):
    """
    以時間為單位的動態調度問題環境，可以跟據poisson機率分布生成增加/減少機器的事件
    """
    # check point data named index
    START_TIME = 0
    NUMBER = 1
    MACHINE_INDEX = 2

    def __init__(self, *args, **kwargs):
        DynamicJSSP.__init__(self, *args, **kwargs)
        self.machine_ratio_threshold: float = 0.2
        self.poisson_mu: int = 0
        self.max_machines_size: dict = {}
        self.check_point_data: dict = {}
        self.check_point_queue: list = []
        self.poisson_generator: scipy.poisson_gen = None
        self.random_generator: np.RandomState = None

    def machine_event(self, machine_type, start_time):
        """
        建立增減機器的事件
        """
        current_size = len(self.machines_map[machine_type])
        max_size = self.max_machines_size[machine_type]
        min_size = 1
        interval_time = self.poisson_generator.rvs(mu=self.poisson_mu)
        prop = (max_size - current_size) / (max_size - min_size)
        if self.random_generator.random() < prop:
            # add machines
            number = self.random_generator.randint(1, max_size - current_size + 1)
            machine_index = None
        else:
            # delete machines
            number = self.random_generator.randint(min_size - current_size, 0)
            machine_index = self.random_generator.choice(current_size, size=-number, replace=False)
        return start_time + interval_time, number, machine_index

    def init_machine_event(self):
        """
        初始化增減機器的事件
        """
        for machine_type, size in self.max_machines_size.items():
            if size > 1:
                self.check_point_data[machine_type] = self.machine_event(machine_type, 0)
        self.check_point_queue = sorted(self.check_point_data.keys(),
                                        key=lambda x: -self.check_point_data[x][TimeBasedJSSP.START_TIME])

    def next_machine_event(self, machine_type):
        """
        下一個增減機器的事件
        """
        start_time = self.check_point_data[machine_type][TimeBasedJSSP.START_TIME]
        self.check_point_data[machine_type] = self.machine_event(machine_type, start_time)
        new_start_time = self.check_point_data[machine_type][TimeBasedJSSP.START_TIME]
        for i, m in enumerate(self.check_point_queue):
            if self.check_point_data[m][TimeBasedJSSP.START_TIME] < new_start_time:
                self.check_point_queue.insert(i, machine_type)
                self.check_point_queue.pop()
                break

    def get_top_machine_event(self):
        """
        獲取最近要發生的增減機器的事件
        """
        machine_type = self.check_point_queue[-1]
        return machine_type, self.check_point_data[machine_type]

    def execute_top_machine_event(self):
        """
        執行最近要發生的增減機器的事件
        """
        machine_type, event_data = self.get_top_machine_event()
        if event_data[TimeBasedJSSP.NUMBER] > 0:
            self.add_machines(
                machine_type, event_data[TimeBasedJSSP.NUMBER], event_data[TimeBasedJSSP.START_TIME])
        else:
            self.delete_machines(
                machine_type, event_data[TimeBasedJSSP.MACHINE_INDEX], event_data[TimeBasedJSSP.START_TIME])
        self.next_machine_event(machine_type)

    def set_poisson_distribution(self, total_duration, jobs, machine_type):
        """
        設置卜瓦松分布參數
        """
        machine_mean_size = np.log2(max(jobs / machine_type, 1)) + 1
        self.poisson_mu = total_duration * machine_mean_size / jobs / machine_type

    def create_num_of_machines(self, jobs, machine_type, type_name):
        """
        建立每種機器種類的機器數量
        """
        mean_value = np.log2(max(jobs / machine_type, 1)) + 1
        min_value = mean_value * (1 - self.machine_ratio_threshold)
        max_value = mean_value * (1 + self.machine_ratio_threshold)
        if min_value < 1:
            max_value = max_value + min_value - 1
            min_value = 1
        max_value += 1
        num_of_machines = {name: np.floor(
            self.random_generator.uniform(min_value, max_value)).astype(int) for name in type_name}
        return num_of_machines

    def set_random_seed(self, random_seed):
        self.poisson_generator = scipy.stats.poisson
        self.poisson_generator.random_state = random_seed
        self.random_generator = np.random.RandomState(random_seed)

    def step(self, action: int):
        """
        執行動作
        """
        feature, reward, done, truncated, info = DynamicJSSP.step(self, action)
        if done or len(self.check_point_queue) == 0:
            return feature, reward, done, truncated, info

        while True:
            _, event_data = self.get_top_machine_event()
            time_limit = event_data[TimeBasedJSSP.START_TIME]

            new_mask_list = []
            for node_id, node_mask in zip(info['node_candidate'], info['node_mask']):
                _, _, _, duration = self.nodes_map[node_id]
                end_time = self.feature_map[node_id][DynamicJSSP.START_TIME] + duration
                new_mask = node_mask or (end_time > time_limit)
                new_mask_list.append(new_mask)

            if all(new_mask_list):
                self.execute_top_machine_event()
                feature = self.get_feature()
                info = self.get_info()
            else:
                info['node_mask'] = new_mask_list
                break
        return feature, reward, done, truncated, info

    def reset(self,
              data: list | np.ndarray = None,
              num_of_machines: dict = None,
              num_of_jobs: int = None,
              num_of_machine_type: int = None,
              random_seed: int = None):
        """
        重設環境
        """
        self.poisson_mu = 1
        self.max_machines_size.clear()
        self.check_point_data.clear()
        self.check_point_queue.clear()
        self.set_random_seed(random_seed)
        if data is None:
            return DynamicJSSP.reset(self, data, num_of_machines)

        if num_of_machine_type is None:
            num_of_jobs = len(data[0])
        if num_of_machine_type is None or num_of_machines is None:
            machine_type_name = set(m for machines in data[1] for m in machines)
            if num_of_machine_type is None:
                num_of_machine_type = len(machine_type_name)
            if num_of_machines is None:
                num_of_machines = self.create_num_of_machines(num_of_jobs, num_of_machine_type, machine_type_name)
        self.set_poisson_distribution(np.sum(data[0]), num_of_jobs, num_of_machine_type)

        ret = DynamicJSSP.reset(self, data, num_of_machines)
        # init machine event
        self.max_machines_size = {k: v for k, v in num_of_machines.items()}
        self.init_machine_event()
        return ret
