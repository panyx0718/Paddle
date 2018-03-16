/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/executor.h"

#include <set>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace framework {
namespace {

std::mutex reduce_mu;
std::vector<std::string> runned;

// block id starts from 0. This id is used to represent the codeblock
// wrapping the first block 0.
int kProgramId = -1;
}  // namespace

struct ExecutorPrepareContext {
  ExecutorPrepareContext(const framework::ProgramDesc& prog, size_t block_id)
      : prog_(prog), block_id_(block_id) {}

  const framework::ProgramDesc& prog_;
  size_t block_id_;
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};


Executor::Executor(const platform::Place& place) : place_(place) {}

static void CreateTensor(Variable* var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::CHANNEL) {
    var->GetMutable<ChannelHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, CHANNEL, RAW]",
        var_type);
  }
}

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type().hash_code() != typeid(float).hash_code() &&
      tensor.type().hash_code() != typeid(double).hash_code()) {
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Tensor %s contains Inf", name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Tensor %s contains NAN", name);
}

void WaitOnPlace(const platform::Place& place) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);
  dev_ctx.Wait();
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id,
                   bool create_local_scope, bool create_vars) {
  platform::RecordBlock b(block_id);
  auto* ctx = Prepare(pdesc, block_id);
  RunPreparedContext(ctx, scope, create_local_scope, create_vars);
  delete ctx;
}

// Check whether the block already has feed operators and feed_holder.
// Return false if the block does not have any feed operators.
// If some feed operators have been prepended to the block, check that
// the info contained in these feed operators matches the feed_targets
// and feed_holder_name. Raise exception when any mismatch is found.
// Return true if the block has feed operators and holder of matching info.
static bool has_feed_operators(
    BlockDesc* block, std::map<std::string, const LoDTensor*>& feed_targets,
    const std::string& feed_holder_name) {
  size_t feed_count = 0;
  for (auto* op : block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      feed_count++;
      PADDLE_ENFORCE_EQ(op->Input("X")[0], feed_holder_name,
                        "Input to feed op should be '%s'", feed_holder_name);
      std::string feed_target_name = op->Output("Out")[0];
      PADDLE_ENFORCE(
          feed_targets.find(feed_target_name) != feed_targets.end(),
          "Feed operator output name '%s' cannot be found in 'feed_targets'",
          feed_target_name);
    }
  }

  if (feed_count > 0) {
    PADDLE_ENFORCE_EQ(
        feed_count, feed_targets.size(),
        "The number of feed operators should match 'feed_targets'");

    // When feed operator are present, so should be feed_holder
    auto var = block->FindVar(feed_holder_name);
    PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                            feed_holder_name);
    PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FEED_MINIBATCH,
                      "'%s' variable should be 'FEED_MINIBATCH' type",
                      feed_holder_name);
  }

  return feed_count > 0;
}

// Check whether the block already has fetch operators and fetch_holder.
// Return false if the block does not have any fetch operators.
// If some fetch operators have been appended to the block, check that
// the info contained in these fetch operators matches the fetch_targets
// and fetch_holder_name. Raise exception when any mismatch is found.
// Return true if the block has fetch operators and holder of matching info.
static bool has_fetch_operators(
    BlockDesc* block, std::map<std::string, LoDTensor*>& fetch_targets,
    const std::string& fetch_holder_name) {
  size_t fetch_count = 0;
  for (auto* op : block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      fetch_count++;
      PADDLE_ENFORCE_EQ(op->Output("Out")[0], fetch_holder_name,
                        "Output of fetch op should be '%s'", fetch_holder_name);
      std::string fetch_target_name = op->Input("X")[0];
      PADDLE_ENFORCE(
          fetch_targets.find(fetch_target_name) != fetch_targets.end(),
          "Fetch operator input name '%s' cannot be found in 'fetch_targets'",
          fetch_target_name);
    }
  }

  if (fetch_count > 0) {
    PADDLE_ENFORCE_EQ(
        fetch_count, fetch_targets.size(),
        "The number of fetch operators should match 'fetch_targets'");

    // When fetch operator are present, so should be fetch_holder
    auto var = block->FindVar(fetch_holder_name);
    PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                            fetch_holder_name);
    PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FETCH_LIST,
                      "'%s' variable should be 'FETCH_LIST' type",
                      fetch_holder_name);
  }

  return fetch_count > 0;
}

void Executor::Run(const ProgramDesc& program, Scope* scope,
                   std::map<std::string, const LoDTensor*>& feed_targets,
                   std::map<std::string, LoDTensor*>& fetch_targets,
                   const std::string& feed_holder_name,
                   const std::string& fetch_holder_name) {
  platform::RecordBlock b(kProgramId);
  auto* copy_program = new ProgramDesc(program);
  auto* global_block = copy_program->MutableBlock(0);

  if (!has_feed_operators(global_block, feed_targets, feed_holder_name)) {
    // create feed_holder variable
    auto* feed_holder = global_block->Var(feed_holder_name);
    feed_holder->SetType(proto::VarType::FEED_MINIBATCH);
    feed_holder->SetPersistable(true);

    // for (auto block : program.Block())

    int i = 0;
    for (auto& feed_target : feed_targets) {
      std::string var_name = feed_target.first;
      VLOG(3) << "feed target's name: " << var_name;

      // prepend feed op
      auto* op = global_block->PrependOp();
      op->SetType(kFeedOpType);
      op->SetInput("X", {feed_holder_name});
      op->SetOutput("Out", {var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  // map the data of feed_targets to feed_holder
  for (auto* op : global_block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      std::string feed_target_name = op->Output("Out")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      SetFeedVariable(scope, *feed_targets[feed_target_name], feed_holder_name,
                      idx);
    }
  }

  if (!has_fetch_operators(global_block, fetch_targets, fetch_holder_name)) {
    // create fetch_holder variable
    auto* fetch_holder = global_block->Var(fetch_holder_name);
    fetch_holder->SetType(proto::VarType::FETCH_LIST);
    fetch_holder->SetPersistable(true);

    int i = 0;
    for (auto& fetch_target : fetch_targets) {
      std::string var_name = fetch_target.first;
      VLOG(3) << "fetch target's name: " << var_name;

      // append fetch op
      auto* op = global_block->AppendOp();
      op->SetType(kFetchOpType);
      op->SetInput("X", {var_name});
      op->SetOutput("Out", {fetch_holder_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  Run(*copy_program, scope, 0, true, true);

  // obtain the data of fetch_targets from fetch_holder
  for (auto* op : global_block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      *fetch_targets[fetch_target_name] =
          GetFetchVariable(*scope, fetch_holder_name, idx);
    }
  }

  delete copy_program;
}

ExecutorPrepareContext* Executor::Prepare(const ProgramDesc& program,
                                          int block_id) {
  auto* ctx = new ExecutorPrepareContext(program, block_id);
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), program.Size());
  auto& block = program.Block(block_id);
  for (auto& op_desc : block.AllOps()) {
    ctx->ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
  return ctx;
}

void Executor::RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                                  bool create_local_scope, bool create_vars) {
  auto& block = ctx->prog_.Block(ctx->block_id_);
  int block_id = block.ID();

  Scope* local_scope = scope;
  if (create_vars) {
    if (create_local_scope) {
      local_scope = &scope->NewScope();
      for (auto& var : block.AllVars()) {
        if (var->Name() == framework::kEmptyVarName) {
          continue;
        }

        if (var->Persistable()) {
          auto* ptr = scope->Var(var->Name());
          CreateTensor(ptr, var->GetType());
          VLOG(3) << "Create Variable " << var->Name()
                  << " global, which pointer is " << ptr;
        } else {
          auto* ptr = local_scope->Var(var->Name());
          CreateTensor(ptr, var->GetType());
          VLOG(3) << "Create Variable " << var->Name()
                  << " locally, which pointer is " << ptr;
        }
      }
    } else {
      for (auto& var : block.AllVars()) {
        auto* ptr = local_scope->Var(var->Name());
        CreateTensor(ptr, var->GetType());
        VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
                << ptr;
      }
    }  // if (create_local_scope)
  }    // if (create_vars)

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto dev_ctx = pool.Get(place_);
  int dev_id_orig = boost::get<platform::CUDAPlace>(place_).GetDeviceId();
  int dev_id = dev_id_orig * 1000 + block_id;
  if (dev_id_orig == 0) {
    std::lock_guard<std::mutex> l2(reduce_mu);
    runned.clear();
  }

  std::mutex to_runs_mu;
  std::deque<OpDesc*> to_runs;
  std::unordered_map<int64_t, OpDesc*> running;
  int64_t cur_id = 0;
  for (auto& op_desc : block.AllOps()) {
    op_desc->Reset(dev_id);
    if (op_desc->IsReady(dev_id)) {
      to_runs.push_back(op_desc);
    }
  }

  std::unordered_map<std::string, OpDesc*> reduces;
  int cur_reduce = 0;

  while (true) {
    int64_t old_id = cur_id;
    cur_id++;
    OpDesc* op_desc = nullptr;
    bool is_all_running = false;
    bool is_too_many_running = false;
    {
      std::lock_guard<std::mutex> l(to_runs_mu);
      if (to_runs.empty()) {
        if (running.empty()) {
          break;
        } else {
          is_all_running = true;
        }
      } else {
        if (running.size() > 50) {
          is_all_running = true;
        } else {
          op_desc = to_runs.front();
          running[old_id] = op_desc;
          to_runs.pop_front();
        }
      }
    }
    if (is_all_running || is_too_many_running) {
      std::this_thread::sleep_for(std::chrono::microseconds(4));
      continue;
    }

    if (op_desc->UniqueName().find("ncclAllReduce") !=
        op_desc->UniqueName().npos) {
      if (dev_id_orig == 0) {
        auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
        // fprintf(stderr, "%s seq_start1 at %d at idx: %lu\n",
        //         op_desc->UniqueName().c_str(), dev_id_orig, runned.size());
        {
          std::lock_guard<std::mutex> l2(reduce_mu);
          runned.push_back(op_desc->UniqueName());
        }
        op->Run(*local_scope, place_);
        // fprintf(stderr, "%s seq_done1\n",
        //         op_desc->UniqueName().c_str());
        std::vector<OpDesc*> nexts = op_desc->GetRunnables(dev_id);
        {
          std::lock_guard<std::mutex> l(to_runs_mu);
          for (int i = 0; i < nexts.size(); ++i) {
            to_runs.push_back(nexts[i]);
          }
          running.erase(old_id);
        }
      } else {
        reduces[op_desc->UniqueName()] = op_desc;
        bool can_run = false;
        {
          std::lock_guard<std::mutex> l2(reduce_mu);
          can_run = cur_reduce < runned.size() &&
                    runned[cur_reduce] == op_desc->UniqueName() &&
                    reduces.find(runned[cur_reduce]) != reduces.end();
        }
        if (can_run) {
          // fprintf(stderr, "to run at idx: %d\n", cur_reduce);
          auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
          // fprintf(stderr, "%s seq_start2 at %d\n",
          //         op_desc->UniqueName().c_str(), dev_id_orig);
          op->Run(*local_scope, place_);
          // fprintf(stderr, "%s seq_done2\n", op_desc->UniqueName().c_str());
          std::vector<OpDesc*> nexts = op_desc->GetRunnables(dev_id);
          {
            std::lock_guard<std::mutex> l(to_runs_mu);
            for (int i = 0; i < nexts.size(); ++i) {
              to_runs.push_back(nexts[i]);
            }
            running.erase(old_id);
          }
          std::lock_guard<std::mutex> l2(reduce_mu);
          cur_reduce++;
        } else {
          std::lock_guard<std::mutex> l(to_runs_mu);
          running.erase(old_id);
          to_runs.push_back(op_desc);
        }
      }
      continue;
    }
    std::thread(
        [this, &to_runs, &to_runs_mu, op_desc, local_scope, dev_ctx, old_id,
            &running, dev_id] {
            OpDesc* desc = op_desc;
            platform::RecordEvent record_event(
                desc->UniqueName(), dev_ctx);
            auto op = paddle::framework::OpRegistry::CreateOp(*desc);
            // fprintf(stderr, "%s start3 at %d\n",
            //         desc->UniqueName().c_str(), dev_id);
            op->Run(*local_scope, place_);
            // fprintf(stderr, "%s done3\n", desc->UniqueName().c_str());

            std::vector<OpDesc*> nexts = desc->GetRunnables(dev_id);
            std::lock_guard<std::mutex> l(to_runs_mu);
            for (int i = 0; i < nexts.size(); ++i) {
              to_runs.push_back(nexts[i]);
            }
            running.erase(old_id);
        }).detach();
  }
  /*
  for (auto& op_desc : block.AllOps()) {
    auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
    VLOG(3) << place_ << " " << op->DebugStringEx(local_scope);
    op->Run(*local_scope, place_);

    std::thread t([&](){
        auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
        VLOG(3) << place_ << " " << op->DebugStringEx(local_scope);
        op->Run(*local_scope, place_);
    });
    t.join();
  }*/

  int64_t no_scheduled = 0;
  for (auto& op_desc : block.AllOps()) {
    if (!op_desc->Scheduled(dev_id)) {
      ++no_scheduled;
      fprintf(stderr, "%s not scheduled at %d\n",
              op_desc->UniqueName().c_str(), dev_id);
    }
  }

  if (create_vars && create_local_scope) {
    scope->DeleteScope(local_scope);
  }
  if (FLAGS_benchmark) {
    VLOG(2) << "-------------------------------------------------------";
    VLOG(2) << "Memory used after deleting local scope: "
            << memory::memory_usage(place_);
    VLOG(2) << "-------------------------------------------------------";
  }
}

}  // namespace framework
}  // namespace paddle
