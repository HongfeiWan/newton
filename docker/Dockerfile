ARG BASE_IMAGE=newton-ubuntu:24.04
FROM ${BASE_IMAGE}

ARG UBUNTU_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG UBUNTU_SUITE=noble

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/workspace/newton/.venv
ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=20

RUN printf 'deb %s %s main universe\n' "${UBUNTU_MIRROR}" "${UBUNTU_SUITE}" > /etc/apt/sources.list \
    && printf 'Acquire::Languages "none";\n' > /etc/apt/apt.conf.d/99no-translations \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    docker.io \
    ffmpeg \
    git \
    iproute2 \
    libatomic1 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    tar \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/newton
COPY . /workspace/newton
COPY docker/wheelhouse /tmp/wheelhouse

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir /tmp/wheelhouse/warp_lang-1.16.0.dev20260707-py3-none-manylinux_2_28_x86_64.whl \
    && python -m pip install --no-cache-dir --find-links /tmp/wheelhouse -e ".[dev]" \
    && python -m pip install --no-cache-dir "isaacteleop[cloudxr]" vosk==0.3.45

CMD ["python", "-m", "newton.tests", "-k", "test_basic.example_basic_shapes"]
