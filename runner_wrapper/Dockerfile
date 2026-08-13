FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

WORKDIR /app/model_repo

COPY requirements.txt /app/requirements.txt

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV VIRTUAL_ENV=/opt/runner
ENV PATH=/opt/runner/bin:${PATH}

RUN python -m venv --system-site-packages ${VIRTUAL_ENV}
RUN grep -vE '^gsplat==' /app/requirements.txt > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation --no-binary=gsplat gsplat==1.5.3
RUN python -c "from gsplat.cuda._backend import _C; assert _C is not None"

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app/model_repo

ENV VIRTUAL_ENV=/opt/runner
ENV PATH=/opt/runner/bin:${PATH}
COPY --from=builder /opt/runner /opt/runner
COPY . /app/model_repo

# If the build context is runner_wrapper/ itself, normalize it
RUN if [ -f server.py ] && [ -f adapter.py ] && [ ! -f runner_wrapper/server.py ]; then \
      mkdir /tmp/runner_wrapper; \
      mv ./* /tmp/runner_wrapper/; \
      mv /tmp/runner_wrapper runner_wrapper; \
    fi

ENV PYTHONPATH=/app/model_repo:/app

RUN python -c "from gsplat.cuda._backend import _C; assert _C is not None"

EXPOSE 58090

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNER_PORT=58090
ENV RUNNER_NAME=fr_iqa
ENV RUNNER_TYPE=evaluator
ENV RUNNER_VERSION=0.1.8
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_IDLE_TIMEOUT_SECONDS=900
ENV RUNNER_ADAPTER=runner_wrapper.fr_iqa_adapter:run_job

CMD ["python", "-m", "runner_wrapper.server"]
