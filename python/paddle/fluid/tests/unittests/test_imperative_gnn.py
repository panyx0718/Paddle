# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import unittest
import numpy as np
import six
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph.base import to_variable


def gen_data():
    pass


class GraphConv(fluid.dygraph.Layer):
    def __init__(self, name_scope, in_features, out_features):
        super(GraphConv, self).__init__(name_scope)

        self._in_features = in_features
        self._out_features = out_features
        self.weight = self.create_parameter(
            attr=None,
            dtype='float32',
            shape=[self._in_features, self._out_features])
        self.bias = self.create_parameter(
            attr=None, dtype='float32', shape=[self._out_features])

    def forward(self, features, adj):
        support = fluid.layers.matmul(features, self.weight)
        # TODO(panyx0718): sparse matmul?
        return fluid.layers.matmul(adj, support) + self.bias


class GCN(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_hidden):
        super(GCN, self).__init__(name_scope)
        self.gc = GraphConv(self.full_name(), num_hidden, 32)
        self.gc2 = GraphConv(self.full_name(), 32, 10)

    def forward(self, x, adj):
        x = fluid.layers.relu(self.gc(x, adj))
        return self.gc2(x, adj)


class TestDygraphGNN(unittest.TestCase):
    def test_gnn_float32(self):
        seed = 90

        startup = fluid.Program()
        startup.random_seed = seed
        main = fluid.Program()
        main.random_seed = seed

        scope = fluid.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            features = fluid.layers.data(
                name='features',
                shape=[1, 100, 50],
                dtype='float32',
                append_batch_size=False)
            # Use selected rows when it's supported.
            adj = fluid.layers.data(
                name='adj',
                shape=[1, 100, 100],
                dtype='float32',
                append_batch_size=False)
            labels = fluid.layers.data(
                name='labels',
                shape=[100, 1],
                dtype='int64',
                append_batch_size=False)

            model = GCN('test_gcn', 50)
            logits = model(features, adj)
            logits = fluid.layers.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = fluid.layers.softmax_with_cross_entropy(logits, labels)
            loss = fluid.layers.reduce_sum(loss)

            adam = AdamOptimizer(learning_rate=1e-3)
            adam.minimize(loss)
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(startup)
            static_loss = exe.run(feed={
                'features': np.zeros(
                    [1, 100, 50], dtype=np.float32),
                'adj': np.zeros(
                    [1, 100, 100], dtype=np.float32),
                'labels': np.zeros(
                    [100, 1], dtype=np.int64)
            },
                                  fetch_list=[loss])[0]

            static_weight = np.array(
                scope.find_var(model.gc.weight.name).get_tensor())

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            features = np.zeros([1, 100, 50], dtype=np.float32)
            # Use selected rows when it's supported.
            adj = np.zeros([1, 100, 100], dtype=np.float32)
            labels = np.zeros([100, 1], dtype=np.int64)

            model = GCN('test_gcn', 50)
            logits = model(to_variable(features), to_variable(adj))
            logits = fluid.layers.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = fluid.layers.softmax_with_cross_entropy(logits,
                                                           to_variable(labels))
            loss = fluid.layers.reduce_sum(loss)
            adam = AdamOptimizer(learning_rate=1e-3)
            adam.minimize(loss)
            self.assertEqual(static_loss, loss._numpy())
            self.assertTrue(
                np.allclose(static_weight, model.gc.weight._numpy()))
            sys.stderr.write('%s %s\n' % (static_loss, loss._numpy()))


if __name__ == '__main__':
    unittest.main()
