#!/usr/bin/env bash
# Build exl3xpu/_inkling2_moe.so (torch library inkling2_moe): ESIMD/DPAS grouped NVFP4/MXFP8 MoE prefill GEMM.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${INKLING2_MOE_FLAGS:-} \
  -I csrc -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ csrc/inkling2_moe_dpas.sycl -x none -o ${INKLING2_MOE_OUT:-exl3xpu/_inkling2_moe.so} \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu 2>&1 | grep -E "error" -A4 | head -60 || true
ls -la ${INKLING2_MOE_OUT:-exl3xpu/_inkling2_moe.so}
