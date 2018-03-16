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

import sys
import numpy as np
import contextlib
from framework import Program, default_main_program, Variable
from . import core

__all__ = [
    'Executor', 'global_scope', 'scope_guard', 'switch_scope', 'fetch_var'
]

g_scope = core.Scope()


def global_scope():
    return g_scope


def switch_scope(scope):
    global g_scope
    ex = g_scope
    g_scope = scope
    return ex


@contextlib.contextmanager
def scope_guard(scope):
    ex = switch_scope(scope)
    yield
    switch_scope(ex)


def as_numpy(tensor):
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError(
            "Some of your featched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly.")
    return np.array(tensor)


def has_feed_operators(block, feed_targets, feed_holder_name):
    """ Check whether the block already has feed operators.

    Return false if the block does not have any feed operators.
    If some feed operators have been prepended to the block, check that
    the info contained in these feed operators matches the feed_targets
    and feed_holder_name. Raise exception when any mismatch is found.
    Return true when the block has feed operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        feed_targets: a dictionary of {feed_target_name: feed_target_data}
        feed_holder_name: the name of the variable that holds the data of
            all feed targets. The type of this feed_holder variable is
            FEED_MINIBATCH, which is essentially vector<LoDTensor>.

    Returns:
        A boolean value that indicates whether a block has feed operators
        that match the info contained in feed_targets and feed_holder_name.
    """

    feed_count = 0
    for op in block.ops:
        if op.desc.type() == 'feed':
            feed_count += 1
            assert op.desc.input('X')[0] == feed_holder_name
            feed_target_name = op.desc.output('Out')[0]
            if feed_target_name not in feed_targets:
                raise Exception("'feed_targets' does not have {} variable".
                                format(feed_target_name))
        else:
            break
    if feed_count > 0 and feed_count != len(feed_targets):
        raise Exception(
            "Feed operators in program desc do not match 'feed_targets'")
    return feed_count > 0


def has_fetch_operators(block, fetch_targets, fetch_holder_name):
    """ Check whether the block already has fetch operators.

    Return false if the block does not have any fetch operators.
    If some fetch operators have been appended to the block, check that
    the info contained in these fetch operators matches the fetch_targets
    and fetch_holder_name. Raise exception when any mismatch is found.
    Return true when the block has fetch operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        fetch_targets: a dictionary of {fetch_target_name: fetch_target_data}
        fetch_holder_name: the name of the variable that holds the data of
            all fetch targets. The type of this fetch_holder variable is
            FETCH_LIST, which is essentially vector<LoDTensor>.

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets and fetch_holder_name.
    """

    fetch_count = 0
    for op in block.ops:
        if op.desc.type() == 'fetch':
            fetch_count += 1
            assert op.desc.output('Out')[0] == fetch_holder_name
            fetch_target_name = op.desc.input('X')[0]
            if fetch_target_name not in [
                    var.desc.name() for var in fetch_targets
            ]:
                raise Exception("'fetch_targets' does not have {} variable".
                                format(fetch_target_name))
            idx = op.desc.attr('col')
            assert fetch_target_name == fetch_targets[idx].desc.name()
    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operators in program desc do not match 'fetch_targets'")
    return fetch_count > 0


def fetch_var(name, scope=None, return_numpy=True):
    """
    Fetch the value of the variable with the given name from the given scope
    Args:
        name(str): name of the variable. Typically, only persistable variables
            can be found in the scope used for running the program.
        scope(core.Scope|None): scope object. It should be the scope where
            you pass to Executor.run() when running your program.
            If None, global_scope() will be used.
        return_numpy(bool): whether convert the tensor to numpy.ndarray
    Returns:
       LodTensor|numpy.ndarray
    """
    assert isinstance(name, str)
    if scope is None:
        scope = global_scope()
    assert isinstance(scope, core.Scope)

    var = global_scope().find_var(name)
    assert var is not None, (
        "Cannot find " + name + " in scope. Perhaps you need to make the"
        " variable persistable by using var.persistable = True in your"
        " program.")
    tensor = var.get_tensor()
    if return_numpy:
        tensor = as_numpy(tensor)
    return tensor


def get_program_cache_key(feed, fetch_list):
    feed_var_names = feed.keys()

    def to_name_str(var):
        if isinstance(var, Variable):
            return var.desc.name()
        elif isinstance(var, str):
            return var
        else:
            raise TypeError(str(var) + " should be Variable or str")

    fetch_var_names = map(to_name_str, fetch_list)

    return str(feed_var_names + fetch_var_names)



def find_loops():
    pass

def analyze_dependency(program):
    output_to_op = dict()
    op_idx = dict()
    ncclInit = None

    idx = 0
    should_print0 = False
    for block in program.blocks:
        for op in block.ops:
            # if 'ncclInit' in op.desc.unique_name():
            #     should_print0 = True
            op.desc.set_unique_name(block.idx)
            op_idx[op.desc.unique_name()] = idx
            idx += 1
            if 'ncclInit' in op.desc.unique_name():
                assert not ncclInit
                ncclInit = op

            for output_name in op.output_names:
                for output_arg in op.output(output_name):
                    if should_print0:
                        sys.stderr.write('ncclInit output %s\n' % output_arg)
                    # if output_arg in output_to_op:
                    #     sys.stderr.write('!!!!!%s duplicated\n' % output_arg)
                    if output_arg == 'fetch':
                        sys.stderr.write('!!!!!%s output from %s\n' %
                                         (output_arg, op.type))
                    if output_arg not in output_to_op:
                        output_to_op[output_arg] = []
                    output_to_op[output_arg].append(op)
            should_print0 = False

    should_print = False
    for block in program.blocks:
        for op in block.ops:
            # if "nccl" in op.desc.unique_name():
            #     sys.stderr.write('nccl: %s\n' % op.desc.unique_name())
            #     should_print = True
            if ncclInit and 'parallel_do_grad' in op.desc.unique_name():
                ncclInit.desc.add_dependent(op.desc)
                sys.stderr.write('%s add dependent %s\n' %
                                 (ncclInit.desc.unique_name(),
                                  op.desc.unique_name()))
            for input_name in op.input_names:
                for input_arg in op.input(input_name):
                    if should_print:
                        sys.stderr.write('nccl input_arg: %s\n' % input_arg)
                    if input_arg not in output_to_op:
                        if should_print:
                            sys.stderr.write('1111\n')
                        # sys.stderr.write(
                        #     '!!!!!%s not output op\n' % input_arg)
                        continue
                    for output_op in output_to_op[input_arg]:
                        if (op_idx[output_op.desc.unique_name()] >=
                            op_idx[op.desc.unique_name()]):
                            if should_print:
                                sys.stderr.write(
                                    '22222 %s %s %d %d\n' % (
                                        op.desc.unique_name(),
                                        output_op.desc.unique_name(),
                                        op_idx[output_op.desc.unique_name()],
                                        op_idx[op.desc.unique_name()]))
                            continue
                        if output_op.block.idx != op.block.idx:
                            if should_print:
                                sys.stderr.write('33333 %d %d\n' %
                                                 (output_op.block.idx,
                                                  op.block.idx))
                            continue
                        # sys.stderr.write('adding %s depends on %s\n' % (
                        #     op.desc.unique_name(),
                        #     output_to_op[input_arg].desc.unique_name()))
                        output_op.desc.add_dependent(op.desc)
            should_print = False
    """        
    op_nexts = dict()
    op_previous = dict()
    for block in program.blocks:
        for op in block.ops:
            nexts = op.desc.all_dep_ops()
            op_nexts[op.desc.unique_name()] = nexts
            for n in nexts:
                if n not in op_previous:
                    op_previous[n] = []
                op_previous[n].append(op.desc.unique_name())
    for op, previous in op_previous.iteritems():
        sys.stderr.write('%s previous %s\n' % (op, ','.join(previous)))
    """
    # sys.stderr.write('%s' % program)


class Executor(object):
    def __init__(self, places):
        if not isinstance(places, list) and not isinstance(places, tuple):
            places = [places]

        act_places = []
        for each in places:
            p = core.Place()
            p.set_place(each)
            act_places.append(p)

        # TODO(dzhwinter) : only use the first place
        self.executor = core.Executor(act_places[0])
        self.places = places
        self.program_caches = dict()
        self._analyzed = set()

    def aslodtensor(self, data):
        def accumulate(data):
            if not isinstance(data, list):
                return 1
            return sum([accumulate(sub) for sub in data])

        def parselod(data):
            seq_lens = [accumulate(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            return lod

        assert len(self.places) != 0
        if not isinstance(data, list):
            # pure tensor case
            tensor = core.LoDTensor()
            tensor.set(data, self.places[0])
            return tensor
        else:
            raise RuntimeError("Current implementation lacks unittests")
            # lodtensor case
            lod = []
            if not isinstance(data[0], list):
                lod.append(parselod(data))
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            else:
                while isinstance(data[0], list):
                    lod.append(parselod(seq))
                    flattened_data = [item for seq in data for item in seq]
                    data = flattened_data
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            flattened_data = flattened_data.reshape([len(flattened_data), 1])
            tensor = core.LoDTensor()
            tensor.set(flattened_data, self.places[0])
            tensor.set_lod(lod)
            return tensor

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
            use_program_cache=False):
        """ Run program by this Executor. Feed data by feed map, fetch result by fetch_list.

        Python executor takes a program, add feed operators and fetch operators to this program according
        to feed map and fetch_list. Feed map provides input data for the program. fetch_list provides
        the variables(or names) that user want to get after program run. Note: the executor will run all
        operators in the program but not only the operators dependent by the fetch_list

        :param program: the program that need to run, if not provied, then default_main_program will be used.
        :param feed: feed variable map, e.g. {"image": ImageData, "label": LableData}
        :param fetch_list: a list of variable or variable names that user want to get, run will return them according
        to this list.
        :param feed_var_name: the name for the input variable of feed Operator.
        :param fetch_var_name: the name for the output variable of feed Operator.
        :param scope: the scope used to run this program, you can switch it to different scope. default is global_scope
        :param return_numpy: if convert the fetched tensor to numpy
        :param use_program_cache: set use_program_cache to true if program not changed compare to the last step.
        :return: result according to fetch_list.
        """
        if feed is None:
            feed = {}
        if not isinstance(feed, dict):
            raise TypeError("feed should be a map")
        if fetch_list is None:
            fetch_list = []

        if program is None:
            program = default_main_program()

        if not isinstance(program, Program):
            raise TypeError()

        if scope is None:
            scope = global_scope()

        program_cache = None
        program_cache_key = get_program_cache_key(feed, fetch_list)

        if use_program_cache:
            # find program cache by cache_key
            program_cache = self.program_caches.get(program_cache_key, None)
            # TODO(qiao): Should check program_cache and program are exactly the same.
        else:
            self.program_caches.pop(program_cache_key, None)

        if program_cache is None:
            program_cache = program.clone()


            if use_program_cache:
                self.program_caches[program_cache_key] = program_cache

            global_block = program_cache.global_block()

            if feed_var_name in global_block.vars:
                feed_var = global_block.var(feed_var_name)
            else:
                feed_var = global_block.create_var(
                    name=feed_var_name,
                    type=core.VarDesc.VarType.FEED_MINIBATCH,
                    persistable=True)

            if fetch_var_name in global_block.vars:
                fetch_var = global_block.var(fetch_var_name)
            else:
                fetch_var = global_block.create_var(
                    name=fetch_var_name,
                    type=core.VarDesc.VarType.FETCH_LIST,
                    persistable=True)

            # prepend feed operators
            if not has_feed_operators(global_block, feed, feed_var_name):
                for i, name in enumerate(feed):
                    out = global_block.var(name)
                    global_block.prepend_op(
                        type='feed',
                        inputs={'X': [feed_var]},
                        outputs={'Out': [out]},
                        attrs={'col': i})

            # append fetch_operators
            if not has_fetch_operators(global_block, fetch_list,
                                       fetch_var_name):
                for i, var in enumerate(fetch_list):
                    assert isinstance(var, Variable) or isinstance(var, str), (
                        "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                    global_block.append_op(
                        type='fetch',
                        inputs={'X': [var]},
                        outputs={'Out': [fetch_var]},
                        attrs={'col': i})

        # feed var to framework
        for op in program_cache.global_block().ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = self.aslodtensor(cur_feed)
                idx = op.desc.attr('col')
                core.set_feed_variable(scope, cur_feed, feed_var_name, idx)
            else:
                break

        if not program_cache in self._analyzed:
            analyze_dependency(program_cache)
            self._analyzed.add(program_cache)
            sys.stderr.write('analyzed a program\n')

        self.executor.run(program_cache.desc, scope, 0, True, True)
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in xrange(len(fetch_list))
        ]
        if return_numpy:
            outs = as_numpy(outs)
        return outs
