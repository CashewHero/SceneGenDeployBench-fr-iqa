FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /app/model_repo

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Compile gsplat for the local RTX 2080 Ti and the deployment RTX A6000.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.6"
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions
RUN python -c "from gsplat.cuda._backend import _C; assert _C is not None"

COPY . /app/model_repo

# If the build context is runner_wrapper/ itself, normalize it
RUN if [ -f server.py ] && [ -f adapter.py ] && [ ! -f runner_wrapper/server.py ]; then \
      mkdir /tmp/runner_wrapper; \
      mv ./* /tmp/runner_wrapper/; \
      mv /tmp/runner_wrapper runner_wrapper; \
    fi

ENV PYTHONPATH=/app/model_repo:/app

EXPOSE 58090

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNER_PORT=58090
ENV RUNNER_NAME=fr_iqa
ENV RUNNER_TYPE=evaluator
ENV RUNNER_VERSION=0.1.0
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_IDLE_TIMEOUT_SECONDS=900
ENV RUNNER_ADAPTER=runner_wrapper.fr_iqa_adapter:run_job

CMD ["python", "-m", "runner_wrapper.server"]
