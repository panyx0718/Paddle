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
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers


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


def SE_ResNeXt(input, label, infer=False):
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
    out = fluid.layers.fc(input=drop, size=1000, act='softmax')
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost


def lenet(data, label):
    conv1 = fluid.layers.conv2d(data, 32, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 50, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=500, act='relu')
    fc2 = fluid.layers.fc(fc1, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=fc2, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss


class CompareParallelExecutorAndParallelDo(unittest.TestCase):
    def model(self, image, label):
        return SE_ResNeXt(image, label)
        # return lenet(image, label)

    def parallel_exe(self, seed, train_inputs):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed

        with fluid.program_guard(main, startup):
            image = fluid.layers.data(
                shape=[-1, 3, 224, 224],
                dtype='float32',
                name='image',
                append_batch_size=False)
            label = fluid.layers.data(
                shape=[-1, 1],
                dtype='int64',
                name='label',
                append_batch_size=False)

            avg_cost = self.model(image, label)
            optimizer = fluid.optimizer.SGD(learning_rate=0.002)
            optimizer.minimize(avg_cost)

            fluid.memory_optimize(fluid.default_main_program())

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            train_exe = fluid.ParallelExecutor(
                loss_name=avg_cost.name, use_cuda=True)
            feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
            losses = []
            for data in train_inputs:
                loss = np.mean(
                    np.array(
                        train_exe.run(fetch_list=[avg_cost.name],
                                      feed=feeder.feed(data))[0]))
                losses.append(loss)
        return losses

    def parallel_do(self, seed, train_inputs):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed

        with fluid.program_guard(main, startup):
            image = fluid.layers.data(
                shape=[-1, 3, 224, 224],
                dtype='float32',
                name='image',
                append_batch_size=False)
            label = fluid.layers.data(
                shape=[-1, 1],
                dtype='int64',
                name='label',
                append_batch_size=False)
            places = fluid.layers.get_places()
            pd = fluid.layers.ParallelDo(places, use_nccl=True)

            with pd.do():
                image_ = pd.read_input(image)
                label_ = pd.read_input(label)
                avg_cost = self.model(image_, label_)
                pd.write_output(avg_cost)

            avg_cost = pd()
            avg_cost = fluid.layers.mean(x=avg_cost)

            optimizer = fluid.optimizer.SGD(learning_rate=0.002)
            optimizer.minimize(avg_cost)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            fluid.memory_optimize(main)

            feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
            losses = []
            for data in train_inputs:
                losses.append(
                    exe.run(main,
                            fetch_list=[avg_cost.name],
                            feed=feeder.feed(data))[0][0])
        return losses

    def test_compare_grad(self):
        seed = 1
        iter = 4

        trn_reader = paddle.batch(flowers.train(), batch_size=2)
        trn_reader_iter = trn_reader()

        train_inputs = []
        for _ in range(iter):
            # train_inputs.append(trn_reader_iter.next())
            # train_inputs.append(
            #    [(np.reshape(np.zeros(shape=[3, 224, 224]), [-1]), 0),
            #     (np.reshape(np.zeros(shape=[3, 224, 224]), [-1]), 0)])
            train_inputs.append(
                [(np.reshape(np.random.rand(3, 224, 224), [-1]), 0),
                 (np.reshape(np.random.rand(3, 224, 224), [-1]), 0)])

        do_losses = self.parallel_do(seed, train_inputs)
        exe_losses = self.parallel_exe(seed, train_inputs)

        sys.stderr.write('loss: %s %s\n' % (do_losses, exe_losses))
        for i in range(iter):
            self.assertTrue(
                np.allclose(
                    do_losses[i], exe_losses[i], atol=1e-8),
                "ParallelDo loss: " + str(do_losses[i]) + "\n ParallelExe loss:"
                + str(exe_losses[i]))


if __name__ == '__main__':
    unittest.main()
