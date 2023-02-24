import numpy as np
import paddle

from lib import utils

paddle.device.set_device("gpu:0")


class LayerParams:
    def __init__(self, rnn_network: paddle.nn.Layer, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = paddle.create_parameter(shape=[*shape], dtype='float32', 
                                               default_initializer=paddle.nn.initializer.XavierNormal())
            self._params_dict[shape] = nn_param
            self._rnn_network.add_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
            return self._params_dict[shape]
        
    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = paddle.create_parameter(shape=[length], dtype='float32', 
                                             default_initializer=paddle.nn.initializer.Constant(value=bias_start))
            self._biases_dict[length] = biases
            self._rnn_network.add_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]
    

class DCGRUCell(paddle.nn.Layer):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = paddle.nn.Tanh if nonlinearity == 'tanh' else paddle.nn.ReLU
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        
        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = paddle.sparse.sparse_coo_tensor(indices=indices.T, values=L.data, shape=[*L.shape])
        return L
    
    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = paddle.nn.functional.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = paddle.reshape(value, (-1, self._num_nodes, output_size))
        r, u = paddle.split(x=value, num_or_sections=self._num_units, axis=-1)
        r = paddle.reshape(r, (-1, self._num_nodes * self._num_units))
        u = paddle.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state
    
    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return paddle.concat([x, x_], axis=0)
    
    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = paddle.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = paddle.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = paddle.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = paddle.nn.functional.sigmoid(paddle.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value
    
    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = paddle.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = paddle.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = paddle.concat([inputs, state], axis=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.transpose([1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = paddle.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = paddle.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = paddle.sparse.matmul(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * paddle.sparse.matmul(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = paddle.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.transpose([3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
        x = paddle.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = paddle.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return paddle.reshape(x, [batch_size, self._num_nodes * output_size])