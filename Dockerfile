FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /opt/triagetier

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY triagetier ./triagetier
COPY settings ./settings

RUN pip install --upgrade pip \
    && pip install -e ".[gpu,imaging]"

ENTRYPOINT ["python", "-m", "triagetier.toolchain.run"]
CMD ["experiment=main"]
