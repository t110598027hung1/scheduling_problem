import pytest
import torch
import random
from env.dynamic_env import *


def csr_convert_tensor(matrix):
    return torch.sparse_csr_tensor(
        crow_indices=torch.tensor(matrix.crow_indices),
        col_indices=torch.tensor(matrix.col_indices),
        values=torch.ones(len(matrix.col_indices)),
        dtype=torch.int32).to_dense()


def coo_convert_tensor(matrix):
    return torch.sparse_coo_tensor(
        indices=torch.tensor([matrix.row_indices, matrix.col_indices]),
        values=torch.ones(len(matrix.col_indices)),
        size=torch.Size((matrix.indices, matrix.indices)),
        dtype=torch.int32).to_dense()


# You can add more implements of different matrix formats, such as COO format
matrix_fixture = [
    (MatrixCSR(), csr_convert_tensor, 'csr'),
    (MatrixCOO(), coo_convert_tensor, 'coo')
]


@pytest.mark.filterwarnings('ignore::UserWarning')
class TestMatrix:
    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_get_info(self, matrix, convert_tensor, _):
        matrix.clear()
        assert matrix.get_info()['value'] == 0
        matrix.new_full_connect([], [1, 2, 3])
        matrix.new_full_connect([1, 2, 3], [4, 5, 6])
        matrix.new_full_connect([4, 5, 6], [7, 8, 9])
        assert matrix.get_info()['value'] == 27
        matrix.add_connect(7, 3)
        assert matrix.get_info()['value'] == 28
        matrix.delete_node({5})
        assert matrix.get_info()['value'] == 21

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_new_full_connect(self, matrix, convert_tensor, _):
        matrix.clear()
        matrix.new_full_connect([], [1, 2, 3])
        matrix.new_full_connect([1, 2, 3], [4, 5, 6])
        matrix.new_full_connect([4, 5, 6], [7, 8, 9])
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))
        matrix.new_full_connect([], [10, 11])
        matrix.new_full_connect([10, 11], [12])
        matrix.new_full_connect([12], [13, 14, 15])
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_add_connect(self, matrix, convert_tensor, _):
        matrix.clear()
        matrix.new_full_connect([], range(10))
        matrix.add_connect(7, 3)
        matrix.add_connect(1, 2)
        matrix.add_connect(2, 1)
        matrix.add_connect(5, 8)
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_batch_add_connect(self, matrix, convert_tensor, _):
        matrix.clear()
        matrix.new_full_connect([], range(1, 11))
        matrix.batch_add_connect(
            [[7, 8, 1, 2, 3], [4, 5, 6], [9]],
            [[4, 5, 6], [1, 2, 3], [8, 10]])
        expect_matrix = [
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))
        matrix.batch_add_connect(
            [[1, 2, 3, 4, 5], [9]],
            [[7], [1, 2, 3, 4, 5]])
        expect_matrix = [
            [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_batch_add_connects_throw_exception(self, matrix, convert_tensor, _):
        matrix.clear()
        matrix.new_full_connect([], range(1, 11))
        try:
            matrix.batch_add_connect(
                [[7, 8, 1, 2, 3]],
                [[4, 5, 6, 6]])
        except ValueError:
            assert True

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_delete_node(self, matrix, convert_tensor, _):
        matrix.clear()
        matrix.new_full_connect([], [0])
        for i in range(10):
            matrix.new_full_connect([i], [i + 1])
        matrix.delete_node({1, 3, 7, 9})
        matrix.delete_node({5})
        expect_matrix = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))

    @pytest.mark.parametrize('matrix, convert_tensor, _', matrix_fixture)
    def test_integration(self, convert_tensor, matrix, _):
        matrix.clear()
        matrix.new_full_connect([], [1, 2, 3])
        matrix.new_full_connect([2], [4, 5])
        matrix.new_full_connect([4, 5], [6])
        matrix.new_full_connect([6], [7, 8, 9])
        matrix.add_connect(1, 4)
        matrix.batch_add_connect(
            [[4], [2, 3]],
            [[7], [6, 9]])
        matrix.delete_node({8, 9})
        matrix.new_full_connect([7], [10, 11, 12])
        matrix.add_connect(10, 6)
        matrix.batch_add_connect(
            [[10, 12]],
            [[11, 5]])
        matrix.delete_node({2})
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(matrix), torch.tensor(expect_matrix))

    @pytest.mark.parametrize('matrix, _, token', matrix_fixture)
    def test_inject_matrix_correct(self, matrix, _, token):
        assert isinstance(inject_matrix(token), type(matrix))

    def test_inject_matrix_exception(self):
        with pytest.raises(ValueError, match='does not exist'):
            inject_matrix('xxx')
        with pytest.raises(ValueError, match='does not exist'):
            inject_matrix('None')
        with pytest.raises(ValueError, match='does not exist'):
            inject_matrix('matrix')
        with pytest.raises(ValueError, match='does not exist'):
            inject_matrix('#12$6')


class TestDynamicJSSP:
    @pytest.mark.parametrize('_, convert_tensor, token', matrix_fixture)
    def test_integration(self, _, convert_tensor, token):
        env = DynamicJSSP(matrix_type=token)
        jss_data = [
            [[30, 40, 10],
             [20, 30, 30]],
            [[1, 2, 3],
             [3, 1, 2]]
        ]
        num_of_machine = {1: 1, 2: 2, 3: 1}
        # reset
        feature, info = env.reset(jss_data, num_of_machine)
        assert env.num_of_nodes == 8
        assert env.jobs_map == {0: [[0], [1, 2], [3]],
                                1: [[4], [5], [6, 7]]}
        assert env.nodes_map == {0: (0, 0, 0, 30), 1: (0, 1, 1, 40), 2: (0, 1, 2, 40), 3: (0, 2, 3, 10),
                                 4: (1, 0, 3, 20), 5: (1, 1, 0, 30), 6: (1, 2, 1, 30), 7: (1, 2, 2, 30)}
        assert env.machines_map == {1: [0], 2: [1, 2], 3: [3]}
        assert env.machines_to_type == {0: 1, 1: 2, 2: 2, 3: 3}
        assert env.machines_last_node == {0: None, 1: None, 2: None, 3: None}
        assert env.machines_start_time == {0: 0, 1: 0, 2: 0, 3: 0}
        assert env.machine_tasks_map == {
            1: [{'job_id': 0, 'task_indices': 0, 'duration': 30, 'pre': [], 'current': [0], 'post': [1, 2]},
                {'job_id': 1, 'task_indices': 1, 'duration': 30, 'pre': [4], 'current': [5], 'post': [6, 7]}],
            2: [{'job_id': 0, 'task_indices': 1, 'duration': 40, 'pre': [0], 'current': [1, 2], 'post': [3]},
                {'job_id': 1, 'task_indices': 2, 'duration': 30, 'pre': [5], 'current': [6, 7], 'post': []}],
            3: [{'job_id': 0, 'task_indices': 2, 'duration': 10, 'pre': [1, 2], 'current': [3], 'post': []},
                {'job_id': 1, 'task_indices': 0, 'duration': 20, 'pre': [], 'current': [4], 'post': [5]}]
        }
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1]
        ]
        assert torch.equal(convert_tensor(env.matrix), torch.tensor(expect_matrix))
        assert np.array_equal(feature, np.array([
            [0.03, 0.00, 0.], [0.07, 0.03, 0.], [0.07, 0.03, 0.], [0.08, 0.07, 0.],
            [0.02, 0.00, 0.], [0.05, 0.02, 0.], [0.08, 0.05, 0.], [0.08, 0.05, 0.]
        ], dtype=np.single))
        assert info['node_candidate'] == [0, 4]
        assert info['matrix_candidate'] == [0, 4]
        assert info['node_mask'] == [False, False]
        assert env.makespan == 80
        # add machines
        env.add_machines(3, 2)
        assert env.num_of_nodes == 12
        assert env.jobs_map == {0: [[0], [1, 2], [3, 8, 9]],
                                1: [[4, 10, 11], [5], [6, 7]]}
        assert env.nodes_map == {0: (0, 0, 0, 30), 1: (0, 1, 1, 40), 2: (0, 1, 2, 40), 3: (0, 2, 3, 10),
                                 4: (1, 0, 3, 20), 5: (1, 1, 0, 30), 6: (1, 2, 1, 30), 7: (1, 2, 2, 30),
                                 8: (0, 2, 4, 10), 9: (0, 2, 5, 10), 10: (1, 0, 4, 20), 11: (1, 0, 5, 20)}
        assert env.machines_map == {1: [0], 2: [1, 2], 3: [3, 4, 5]}
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
        assert torch.equal(convert_tensor(env.matrix), torch.tensor(expect_matrix))
        assert np.array_equal(env.get_feature(), np.array([
            [0.03, 0.00, 0.], [0.07, 0.03, 0.], [0.07, 0.03, 0.], [0.08, 0.07, 0.],
            [0.02, 0.00, 0.], [0.05, 0.02, 0.], [0.08, 0.05, 0.], [0.08, 0.05, 0.],
            [0.08, 0.07, 0.], [0.08, 0.07, 0.], [0.02, 0.00, 0.], [0.02, 0.00, 0.]
        ], dtype=np.single))
        info = env.get_info()
        assert info['node_candidate'] == [0, 4, 10, 11]
        assert info['matrix_candidate'] == [0, 4, 10, 11]
        assert info['node_mask'] == [False, False, False, False]
        assert env.makespan == 80
        # add jobs
        env.add_jobs([20, 30, 30], [3, 2, 1])
        assert env.num_of_nodes == 18
        assert env.jobs_map == {0: [[0], [1, 2], [3, 8, 9]],
                                1: [[4, 10, 11], [5], [6, 7]],
                                2: [[12, 13, 14], [15, 16], [17]]}
        assert env.nodes_map == {0: (0, 0, 0, 30), 1: (0, 1, 1, 40), 2: (0, 1, 2, 40),
                                 3: (0, 2, 3, 10), 4: (1, 0, 3, 20), 5: (1, 1, 0, 30),
                                 6: (1, 2, 1, 30), 7: (1, 2, 2, 30), 8: (0, 2, 4, 10),
                                 9: (0, 2, 5, 10), 10: (1, 0, 4, 20), 11: (1, 0, 5, 20),
                                 12: (2, 0, 3, 20), 13: (2, 0, 4, 20), 14: (2, 0, 5, 20),
                                 15: (2, 1, 1, 30), 16: (2, 1, 2, 30), 17: (2, 2, 0, 30)}
        assert env.machines_map == {1: [0], 2: [1, 2], 3: [3, 4, 5]}
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        ]
        assert torch.equal(convert_tensor(env.matrix), torch.tensor(expect_matrix))
        assert np.array_equal(env.get_feature(), np.array([
            [0.03, 0.00, 0.], [0.07, 0.03, 0.], [0.07, 0.03, 0.], [0.08, 0.07, 0.],
            [0.02, 0.00, 0.], [0.05, 0.02, 0.], [0.08, 0.05, 0.], [0.08, 0.05, 0.],
            [0.08, 0.07, 0.], [0.08, 0.07, 0.], [0.02, 0.00, 0.], [0.02, 0.00, 0.],
            [0.02, 0.00, 0.], [0.02, 0.00, 0.], [0.02, 0.00, 0.], [0.05, 0.02, 0.],
            [0.05, 0.02, 0.], [0.08, 0.05, 0.]
        ], dtype=np.single))
        info = env.get_info()
        assert info['node_candidate'] == [0, 4, 10, 11, 12, 13, 14]
        assert info['matrix_candidate'] == [0, 4, 10, 11, 12, 13, 14]
        assert info['node_mask'] == [False, False, False, False, False, False, False]
        # delete machines
        env.delete_machines(3, [1])
        assert env.num_of_nodes == 15
        assert env.jobs_map == {0: [[0], [1, 2], [3, 9]],
                                1: [[4, 11], [5], [6, 7]],
                                2: [[12, 14], [15, 16], [17]]}
        assert env.machines_map == {1: [0], 2: [1, 2], 3: [3, 5]}
        expect_matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        ]
        assert torch.equal(convert_tensor(env.matrix), torch.tensor(expect_matrix))
        assert np.array_equal(env.get_feature(), np.array([
            [0.03, 0.00, 0.], [0.07, 0.03, 0.], [0.07, 0.03, 0.], [0.08, 0.07, 0.],
            [0.02, 0.00, 0.], [0.05, 0.02, 0.], [0.08, 0.05, 0.], [0.08, 0.05, 0.],
            [0.08, 0.07, 0.], [0.02, 0.00, 0.], [0.02, 0.00, 0.], [0.02, 0.00, 0.],
            [0.05, 0.02, 0.], [0.05, 0.02, 0.], [0.08, 0.05, 0.]
        ], dtype=np.single))
        info = env.get_info()
        assert info['node_candidate'] == [0, 4, 11, 12, 14]
        assert info['matrix_candidate'] == [0, 4, 9, 10, 11]
        assert info['node_mask'] == [False, False, False, False, False]
        assert env.makespan == 80

    @pytest.mark.parametrize('_, convert_tensor, token', matrix_fixture)
    def test_step_and_done(self, _, convert_tensor, token):
        env = DynamicJSSP(matrix_type=token)
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

        def simulation():
            env.reset(jss_data, num_of_machine)
            for index, action in enumerate(actions):
                feature, reward, done, _, _ = env.step(action)
                if expected_features:
                    assert np.array_equal(feature, np.array(expected_features[index], dtype=np.single))
                assert reward == expected_rewards[index]
                assert done == expected_dones[index]
            assert env.makespan == expected_makespan

        actions = [0, 12, 6, 15, 9, 1, 3, 10, 17]
        expected_features = [
            [[0.03, 0.03, 1.0],
             [0.07, 0.03, 0.0],
             [0.07, 0.03, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.03, 0.0],
             [0.08, 0.05, 0.0],
             [0.08, 0.05, 0.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.02, 0.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.05, 0.0]],
            [[0.03, 0.03, 1.0],
             [0.07, 0.03, 0.0],
             [0.07, 0.03, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.02, 0.02, 0.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.03, 0.0],
             [0.08, 0.05, 0.0],
             [0.08, 0.05, 0.0],
             [0.02, 0.02, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.02, 0.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.05, 0.0]],
            [[0.03, 0.03, 1.0],
             [0.07, 0.03, 0.0],
             [0.07, 0.03, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.04, 0.04, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.04, 0.0],
             [0.1, 0.07, 0.0],
             [0.1, 0.07, 0.0],
             [0.02, 0.04, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.02, 0.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.05, 0.0]],
            [[0.03, 0.03, 1.0],
             [0.07, 0.05, 0.0],
             [0.07, 0.03, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.04, 0.04, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.04, 0.0],
             [0.1, 0.07, 0.0],
             [0.1, 0.07, 0.0],
             [0.02, 0.04, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.05, 1.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.05, 0.0]],
            [[0.03, 0.07, 1.0],
             [0.07, 0.05, 0.0],
             [0.07, 0.03, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.08, 0.07, 0.0],
             [0.04, 0.04, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.07, 1.0],
             [0.1, 0.07, 0.0],
             [0.1, 0.07, 0.0],
             [0.02, 0.04, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.05, 1.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.07, 0.0]],
            [[0.03, 0.07, 1.0],
             [0.09, 0.09, 1.0],
             [0.09, 0.05, 0.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.09, 0.0],
             [0.04, 0.04, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.07, 1.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.07, 0.0],
             [0.02, 0.04, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.09, 1.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.07, 0.0]],
            [[0.03, 0.07, 1.0],
             [0.09, 0.09, 1.0],
             [0.09, 0.05, 0.0],
             [0.1, 0.1, 1.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.09, 0.0],
             [0.04, 0.1, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.07, 1.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.07, 0.0],
             [0.02, 0.1, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.09, 1.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.07, 0.0]],
            [[0.03, 0.07, 1.0],
             [0.09, 0.12, 1.0],
             [0.09, 0.05, 0.0],
             [0.1, 0.1, 1.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.09, 0.0],
             [0.04, 0.1, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.07, 1.0],
             [0.12, 0.12, 1.0],
             [0.12, 0.09, 0.0],
             [0.02, 0.1, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.12, 1.0],
             [0.05, 0.02, 0.0],
             [0.08, 0.07, 0.0]],
            [[0.03, 0.1, 1.0],
             [0.09, 0.12, 1.0],
             [0.09, 0.05, 0.0],
             [0.1, 0.1, 1.0],
             [0.1, 0.09, 0.0],
             [0.1, 0.09, 0.0],
             [0.04, 0.1, 1.0],
             [0.04, 0.02, 0.0],
             [0.04, 0.02, 0.0],
             [0.07, 0.1, 1.0],
             [0.12, 0.12, 1.0],
             [0.12, 0.09, 0.0],
             [0.02, 0.1, 1.0],
             [0.02, 0.0, 0.0],
             [0.02, 0.0, 0.0],
             [0.05, 0.12, 1.0],
             [0.05, 0.02, 0.0],
             [0.1, 0.1, 1.0]]
        ]
        expected_rewards = [0, 0, -20, 0, 0, 0, 0, -20, 0]
        expected_makespan = 120
        simulation()
        actions = [0, 12, 7, 15, 9, 2, 5, 11, 17]
        expected_features = None
        expected_rewards = [0, 0, 0, 0, -10, 0, 0, -10, 0]
        expected_makespan = 100
        simulation()
        actions = [0, 12, 7, 15, 9, 2, 5, 10, 17]
        expected_features = None
        expected_rewards = [0, 0, 0, 0, -10, 0, 0, 0, 0]
        expected_makespan = 90
        simulation()

    @pytest.mark.parametrize('_, convert_tensor, token', matrix_fixture)
    def test_get_pool_mask(self, _, convert_tensor, token):
        env = DynamicJSSP(matrix_type=token)
        jss_data = [
            [[30, 40, 10],
             [20, 30, 30]],
            [[1, 2, 3],
             [3, 1, 2]]
        ]
        num_of_machine = {1: 1, 2: 2, 3: 3}
        env.reset(jss_data, num_of_machine)
        expected_jobs_map = {0: [[0], [1, 2], [3, 4, 5]], 1: [[6, 7, 8], [9], [10, 11]]}
        assert env.jobs_map == expected_jobs_map

        env.step(0)
        expected_mask = [False, False, False, False, False, False, False, False, False, False, False, False]
        assert env.get_pool_mask() == expected_mask

        env.step(6)
        expected_mask = [False, False, False, False, False, False, False, True, True, False, False, False]
        assert env.get_pool_mask() == expected_mask

        env.step(2)
        expected_mask = [False, True, False, False, False, False, False, True, True, False, False, False]
        assert env.get_pool_mask() == expected_mask

        env.add_machines(1, 1, 100)
        expected_jobs_map = {0: [[0, 12], [1, 2], [3, 4, 5]], 1: [[6, 7, 8], [9, 13], [10, 11]]}
        expected_mask = [False, True, False, False, False, False, False, True, True, False, False, False, True, False]
        assert env.jobs_map == expected_jobs_map
        assert env.get_pool_mask() == expected_mask

        env.step(13)
        expected_mask = [False, True, False, False, False, False, False, True, True, True, False, False, True, False]
        assert env.get_pool_mask() == expected_mask

        env.delete_machines(3, [1], 200)
        expected_jobs_map = {0: [[0, 12], [1, 2], [3, 5]], 1: [[6, 8], [9, 13], [10, 11]]}
        expected_mask = [False, True, False, False, False, False, True, True, False, False, True, False]
        assert env.jobs_map == expected_jobs_map
        assert env.get_pool_mask() == expected_mask

        env.step(5)
        expected_mask = [False, True, False, True, False, False, True, True, False, False, True, False]
        assert env.get_pool_mask() == expected_mask

    @pytest.mark.parametrize('_, convert_tensor, token', matrix_fixture)
    def test_reset_num_of_machines_is_none(self, _, convert_tensor, token):
        env = DynamicJSSP(matrix_type=token)
        jss_data = [
            [[30, 40, 10],
             [20, 30, 30],
             [20, 30, 30]],
            [[1, 2, 3],
             [3, 1, 2],
             [3, 2, 1]]
        ]
        env.reset(jss_data)
        assert env.machines_map == {1: [0], 2: [1], 3: [2]}

    @pytest.mark.parametrize('_, convert_tensor, token', matrix_fixture)
    def test_delete_machines_throw_exception(self, _, convert_tensor, token):
        env = DynamicJSSP(matrix_type=token)
        jss_data = [
            [[30, 40, 10],
             [20, 30, 30],
             [20, 30, 30]],
            [[1, 2, 3],
             [3, 1, 2],
             [3, 2, 1]]
        ]
        env.reset(jss_data)
        try:
            env.delete_machines(1, [0], 100)
            assert False
        except ValueError:
            assert True


class TestTimeBasedJSSP:
    def test_step_and_done(self):
        env = TimeBasedJSSP()
        jss_data = [
            [[30, 40, 20],
             [20, 10, 60],
             [30, 20, 10],
             [10, 30, 30]],
            [[1, 2, 3],
             [2, 1, 3],
             [3, 2, 1],
             [3, 1, 2]]
        ]
        actions = [0, 5, 10, 7, 16, 12, 18, 14, 2, 19, 9, 20]
        expected_num_of_nodes = [20, 20, 20, 20, 16, 12, 12, 20, 20, 20, 20, 20]
        expected_machines_map = [
            {1: [0, 1], 2: [2], 3: [3, 4]},
            {1: [0, 1], 2: [2], 3: [3, 4]},
            {1: [0, 1], 2: [2], 3: [3, 4]},
            {1: [0, 1], 2: [2], 3: [3, 4]},
            {1: [0, 1], 2: [2], 3:    [4]},
            {1:    [1], 2: [2], 3:    [4]},
            {1:    [1], 2: [2], 3:    [4]},
            {1: [1, 6], 2: [2], 3: [4, 5]},
            {1: [1, 6], 2: [2], 3: [4, 5]},
            {1: [1, 6], 2: [2], 3: [4, 5]},
            {1: [1, 6], 2: [2], 3: [4, 5]},
            {1: [1, 6], 2: [2], 3: [4, 5]}
        ]
        expected_machines_start_time = [
            {0:  30, 1:   0, 2:   0, 3:   0, 4:   0},
            {0:  30, 1:   0, 2:  20, 3:   0, 4:   0},
            {0:  30, 1:   0, 2:  20, 3:  30, 4:   0},
            {0:  30, 1:  30, 2:  20, 3:  30, 4:   0},
            {0:  30, 1:  30, 2:  20, 4:  10},
            {1:  30, 2:  50, 4:  10},
            {1:  60, 2:  50, 4:  10},
            {1:  70, 2:  50, 4:  10, 5:  80, 6:  84},
            {1:  70, 2:  90, 4:  10, 5:  80, 6:  84},
            {1:  70, 2: 120, 4:  10, 5:  80, 6:  84},
            {1:  70, 2: 120, 4:  90, 5:  80, 6:  84},
            {1:  70, 2: 120, 4:  90, 5: 110, 6:  84}
        ]
        expected_machine_event = [
            {1: (50, -1, [0]), 3: (38, -1, [0])},
            {1: (50, -1, [0]), 3: (38, -1, [0])},
            {1: (50, -1, [0]), 3: (38, -1, [0])},
            {1: (50, -1, [0]), 3: (38, -1, [0])},
            {1: (50, -1, [0]), 3: (80, 1, None)},
            {1: (84, 1, None), 3: (80, 1, None)},
            {1: (84, 1, None), 3: (80, 1, None)},
            {1: (128, -1, [0]), 3: (127, -1, [0])},
            {1: (128, -1, [0]), 3: (127, -1, [0])},
            {1: (128, -1, [0]), 3: (127, -1, [0])},
            {1: (128, -1, [0]), 3: (127, -1, [0])},
            {1: (128, -1, [0]), 3: (127, -1, [0])}
        ]
        expected_dones = [False, False, False, False, False, False, False, False, False, False, False, True]
        expected_rewards = [0, 0, 0, 0, 0, 0, 0, 0, -20, -10, 0, 0]
        expected_makespan = 120
        env.reset(jss_data, random_seed=200)
        for index, action in enumerate(actions):
            feature, reward, done, _, info = env.step(action)
            assert env.num_of_nodes == expected_num_of_nodes[index]
            assert env.machines_map == expected_machines_map[index]
            assert env.machines_start_time == expected_machines_start_time[index]
            assert reward == expected_rewards[index]
            assert done == expected_dones[index]
            assert env.check_point_data == expected_machine_event[index]
        assert env.makespan == expected_makespan

    def test_random_choice(self):
        num_of_jobs = 10
        num_of_machines = 5
        random.seed(200)
        np.random.seed(200)
        jss_data = [
            [[random.randint(1, 99) for _ in range(num_of_machines)] for _ in range(num_of_jobs)],
            [random.sample(range(1, num_of_machines + 1), num_of_machines) for _ in range(num_of_jobs)]
        ]
        done = False
        env = TimeBasedJSSP()
        _, info = env.reset(jss_data)
        while not done:
            candidate = [node for node, mask in zip(info['node_candidate'], info['node_mask']) if not mask]
            action = random.choice(candidate)
            _, _, done, _, info = env.step(action)

        machines = {}
        machines_info = env.machines_info
        jobs_info = env.jobs_info
        # assert schedule equivalent to jss_data
        for job_index, job in jobs_info.items():
            for task, data_duration, data_machine_type in zip(job, jss_data[0][job_index], jss_data[1][job_index]):
                schedule_duration = task['end'] - task['start']
                schedule_machine_type = task['machine_type']
                assert schedule_duration == data_duration
                assert schedule_machine_type == data_machine_type
        # assert conjunction
        for job in jobs_info.values():
            for pre_task, post_task in zip(job[:-1], job[1:]):
                assert post_task['start'] >= pre_task['end']
            for task in job:
                machine_index = (task['machine_type'], task['machine_id'])
                if machine_index not in machines:
                    machines[machine_index] = []
                machines[machine_index].append(task)
        # assert disjunction
        for machine_index, tasks in machines.items():
            machine = machines_info[machine_index[0]][machine_index[1]]
            tasks = sorted(tasks, key=lambda x: x['start'])
            for pre_task, post_task in zip(tasks[:-1], tasks[1:]):
                assert post_task['start'] >= pre_task['end']
            assert tasks[0]['start'] >= machine['start']
            if 'end' in machine:
                assert tasks[0]['end'] <= machine['end']

    def test_reset_with_data_is_none(self):
        super_env = DynamicJSSP()
        super_env.reset()
        sub_env = TimeBasedJSSP()
        sub_env.reset()
        for attribute, value in super_env.__dict__.items():
            comparison_value = sub_env.__dict__[attribute]
            print(type(value), attribute)
            if attribute == 'normalize_coefficient':
                assert np.array_equal(comparison_value, value)
            elif attribute == 'matrix':
                assert type(comparison_value) == type(value)
            else:
                assert comparison_value == value

    def test_reset_with_same_num_of_job_and_machines(self):
        num_of_jobs = num_of_machines = 3
        jss_data = [
            [[random.randint(1, 99) for _ in range(num_of_machines)] for _ in range(num_of_jobs)],
            [random.sample(range(1, num_of_machines + 1), num_of_machines) for _ in range(num_of_jobs)]
        ]
        env = TimeBasedJSSP()
        env.reset(jss_data)
        assert env.max_machines_size == {1: 1, 2: 1, 3: 1}
