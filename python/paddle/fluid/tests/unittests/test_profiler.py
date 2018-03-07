#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.layers as layers
import paddle.fluid.core as core


class TestProfiler(unittest.TestCase):
    def net_profiler(self, state, profile_path='/tmp/profile'):
        enable_if_gpu = state == 'GPU' or state == "All"
        if enable_if_gpu and not core.is_compiled_with_cuda():
            return
        startup_program = fluid.Program()
        main_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            image = fluid.layers.data(name='x', shape=[784], dtype='float32')
            predict = fluid.layers.fc(input=image, size=10, act='relu')
            label = fluid.layers.data(name='y', shape=[1], dtype='int64')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)
            accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

        optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opts = optimizer.minimize(avg_cost, startup_program=startup_program)

        place = fluid.CPUPlace() if state == 'CPU' else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        accuracy.reset(exe)
        with profiler.profiler(state, 'total', profile_path) as prof:
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                x = np.random.random((32, 784)).astype("float32")
                y = np.random.randint(0, 10, (32, 1)).astype("int64")

                outs = exe.run(main_program,
                               feed={'x': x,
                                     'y': y},
                               fetch_list=[avg_cost] + accuracy.metrics)
                acc = np.array(outs[1])
                pass_acc = accuracy.eval(exe)

    # def test_cpu_profiler(self):
    #    self.net_profiler('CPU')

    def test_cuda_profiler(self):
        self.net_profiler('GPU')

    # def test_all_profiler(self):
    #     self.net_profiler('All', '/tmp/profile_out')
    #     with open('/tmp/profile_out', 'r') as f:
    #         self.assertGreater(len(f.read()), 0)


if __name__ == '__main__':
    unittest.main()
