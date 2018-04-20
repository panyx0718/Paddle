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
import sys
import time
import numpy as np
import paddle.fluid as fluid


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False):
    cardinality = 64
    reduction_ratio = 16
    depth = [3, 8, 36, 3]
    num_filters = [128, 256, 512, 1024]

    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


class CompareParallelExecutorAndParallelDo(unittest.TestCase):
    def parallel_exe(self, seed, iter_times):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        class_dim = 1000

        with fluid.program_guard(main, startup):
            image = fluid.layers.fill_constant(
                shape=[12, 3, 224, 224], dtype='float32', value=0.0)
            label = fluid.layers.fill_constant(
                shape=[12, 1], dtype='int64', value=0.0)

            out = SE_ResNeXt(input=image, class_dim=class_dim)
            cost = fluid.layers.cross_entropy(input=out, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            optimizer = fluid.optimizer.SGD(learning_rate=0.002)
            optimizer.minimize(avg_cost)

            fluid.memory_optimize(fluid.default_main_program())

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            train_exe = fluid.ParallelExecutor(
                loss_name=avg_cost.name, use_cuda=True)
            losses = []
            for _ in xrange(iter_times):
                loss = np.mean(
                    np.array(train_exe.run(fetch_list=[avg_cost.name])[0]))
                losses.append(loss)
        return losses

    def parallel_do(self, seed, iter_times):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        class_dim = 1000

        with fluid.program_guard(main, startup):
            image = fluid.layers.fill_constant(
                shape=[12, 3, 224, 224], dtype='float32', value=0.0)
            label = fluid.layers.fill_constant(
                shape=[12, 1], dtype='int64', value=0.0)
            places = fluid.layers.get_places()
            pd = fluid.layers.ParallelDo(places, use_nccl=True)

            with pd.do():
                image_ = pd.read_input(image)
                label_ = pd.read_input(label)
                out = SE_ResNeXt(input=image_, class_dim=class_dim)
                cost = fluid.layers.cross_entropy(input=out, label=label_)
                avg_cost = fluid.layers.mean(x=cost)
                accuracy = fluid.layers.accuracy(input=out, label=label_)
                pd.write_output(avg_cost)
                pd.write_output(accuracy)

            avg_cost, accuracy = pd()
            avg_cost = fluid.layers.mean(x=avg_cost)

            optimizer = fluid.optimizer.SGD(learning_rate=0.002)
            optimizer.minimize(avg_cost)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            fluid.memory_optimize(main)

            losses = []
            for _ in xrange(iter_times):
                losses.append(exe.run(main, fetch_list=[avg_cost.name])[0][0])
        return losses

    def test_compare_grad(self):
        seed = 1
        iter = 10
        do_losses = self.parallel_do(seed, iter)
        exe_losses = self.parallel_exe(seed, iter)

        for i in range(iter):
            sys.stderr.write('loss: %s %s\n' % (do_losses[i], exe_losses[i]))
            self.assertTrue(
                np.allclose(
                    do_losses[i], exe_losses[i], atol=1e-8),
                "ParallelDo loss: " + str(do_losses[i]) + "\n ParallelExe loss:"
                + str(exe_losses[i]))


if __name__ == '__main__':
    unittest.main()
