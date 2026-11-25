# exl3xpu: EXL3 inference on Intel Arc (Battlemage) via vLLM XPU.
FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
# Build from the repository root:  docker build -f xpu/docker/Dockerfile -t exl3xpu .
COPY core /opt/trellis-core
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-core
WORKDIR /opt/exl3xpu
COPY xpu/vllm_patches /opt/vllm_patches
RUN cp /opt/vllm_patches/dflash_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/utils.py \
 && cp /opt/vllm_patches/qwen3_dflash.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
COPY xpu/pyproject.toml ./
COPY xpu/exl3xpu ./exl3xpu
COPY xpu/csrc ./csrc
COPY xpu/scripts ./scripts
COPY xpu/models ./models
COPY xpu/bench ./bench
COPY xpu/tests ./tests
ARG EXL3_FLAGS=
ENV EXL3_FLAGS=${EXL3_FLAGS}
RUN bash scripts/build_ext.sh && pip install --no-deps --no-build-isolation -e . \
 && python3 -c "from importlib.metadata import entry_points as e; assert any(x.name=='exl3xpu' for x in e(group='vllm.general_plugins'))"
ENV MODELS_DIR=/models
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
