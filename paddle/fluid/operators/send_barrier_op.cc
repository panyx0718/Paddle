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

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class SendBarrierOp : public framework::OperatorBase {
 public:
  SendBarrierOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    std::vector<std::string> eps = Attr<std::vector<std::string>>("endpoints");
    bool sync_mode = Attr<bool>("sync_mode");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);
    // For profiling
    platform::RecordEvent record_event(Type(), &ctx);

    std::unique_ptr<platform::RecordEvent> r1(
        new platform::RecordEvent("CreateClient", nullptr));
    auto rpc_client = detail::RPCClient::GetInstance();
    r1.reset(nullptr);

    r1.reset(
        new platform::RecordEvent("ClientWait", nullptr));
    // need to wait before sending send_barrier message
    PADDLE_ENFORCE(rpc_client->Wait());
    r1.reset(nullptr);

    if (sync_mode) {
      for (auto& ep : eps) {
        platform::RecordEvent record_event2("Send_" + ep, &ctx);
        VLOG(3) << "send barrier, ep: " << ep;
        rpc_client->AsyncSendBatchBarrier(ep);
      }
      platform::RecordEvent record_event3("send_barrier_wait", &ctx);
      PADDLE_ENFORCE(rpc_client->Wait());
    }
  }
};

class SendBarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddComment(R"DOC(
SendBarrier operator

This operator will send a send barrier signal to list_and_serv op, so that
the Parameter Server would knew all variables have been sent.
)DOC");

    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<bool>("sync_mode", "work in sync_mode or not").SetDefault(true);
  }
};

class SendBarrierOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send_barrier, ops::SendBarrierOp,
                  paddle::framework::EmptyGradOpMaker, ops::SendBarrierOpMaker,
                  ops::SendBarrierOpShapeInference);
