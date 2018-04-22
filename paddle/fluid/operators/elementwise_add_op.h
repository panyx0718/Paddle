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

#pragma once

#include "paddle/fluid/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          AddFunctor<T>(), z);

    /*
    std::vector<T> xv;
    framework::TensorToVector(*x, ctx.device_context(), &xv);
    double total = 0.0;
    for (T v : xv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total += static_cast<double>(v1);
    }
    fprintf(stderr, "fx: %f\n", static_cast<double>(total));
    std::vector<T> yv;
    framework::TensorToVector(*y, ctx.device_context(), &yv);
    double total01 = 0.0;
    for (T v : yv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total01 += static_cast<double>(v1);
    }
    fprintf(stderr, "fy: %f\n", static_cast<double>(total01));
    std::vector<T> outv;
    framework::TensorToVector(*z, ctx.device_context(), &outv);
    double total02 = 0.0;
    for (T v : outv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total02 += static_cast<double>(v1);
    }
    fprintf(stderr, "fz: %f\n", static_cast<double>(total02));*/
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");

    std::vector<T> xv;
    framework::TensorToVector(*x, ctx.device_context(), &xv);
    T total = 0.0;
    for (T v : xv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total += v1;
    }
    fprintf(stderr, "x: %f\n", static_cast<double>(total));
    std::vector<T> yv;
    framework::TensorToVector(*y, ctx.device_context(), &yv);
    T total01 = 0.0;
    for (T v : yv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total01 += v1;
    }
    fprintf(stderr, "y: %f\n", static_cast<double>(total01));
    std::vector<T> outv;
    framework::TensorToVector(*out, ctx.device_context(), &outv);
    T total02 = 0.0;
    for (T v : outv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total02 += v1;
    }
    fprintf(stderr, "out: %f\n", static_cast<double>(total02));

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, IdentityGrad<T>, IdentityGrad<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
        IdentityGrad<T>());

    std::vector<T> doutv;
    framework::TensorToVector(*dout, ctx.device_context(), &doutv);
    T total03 = 0.0;
    for (T v : doutv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total03 += v1;
    }
    fprintf(stderr, "dout: %f\n", static_cast<double>(total03));
    std::vector<T> dxv;
    framework::TensorToVector(*dx, ctx.device_context(), &dxv);
    T total2 = 0.0;
    for (T v : dxv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total2 += v1;
    }
    fprintf(stderr, "dx: %f\n", static_cast<double>(total2));
    std::vector<T> dyv;
    framework::TensorToVector(*dy, ctx.device_context(), &dyv);
    T total3 = 0.0;
    for (T v : dyv) {
      T v1 = v;
      if (v1 < 0) {
        v1 = -v1;
      }
      total3 += v1;
    }
    fprintf(stderr, "dy: %f\n\n", static_cast<double>(total3));
  }
};

}  // namespace operators
}  // namespace paddle
