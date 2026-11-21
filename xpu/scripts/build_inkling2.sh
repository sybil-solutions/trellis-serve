#!/usr/bin/env bash
# Build exl3xpu/_inkling2.so (torch library inkling2_xpu): NVFP4/MXFP8 MoE GEMV + dequant for Inkling 2.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${INKLING2_FLAGS:-} \
  -I csrc -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ csrc/inkling2_ops.sycl -x none -o ${INKLING2_OUT:-exl3xpu/_inkling2.so} \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
ls -la ${INKLING2_OUT:-exl3xpu/_inkling2.so}
