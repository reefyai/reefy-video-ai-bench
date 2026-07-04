FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS nvidia

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-nvidia.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-nvidia.txt

COPY raw_video_detector_bench ./raw_video_detector_bench
COPY labels ./labels
COPY models ./models

ENTRYPOINT ["python3", "-m", "raw_video_detector_bench.bench"]


FROM ubuntu:24.04 AS intel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        python3 \
        python3-pip \
        ffmpeg \
        pciutils \
        vainfo \
        clinfo \
    && install -d -m 0755 /usr/share/keyrings \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        intel-level-zero-gpu \
        intel-opencl-icd \
        intel-media-va-driver-non-free \
        libva2 \
        libva-drm2 \
        libva-x11-2 \
        libva-wayland2 \
        libmfxgen1 \
        libvpl2 \
        libze1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-intel.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-intel.txt

COPY raw_video_detector_bench ./raw_video_detector_bench
COPY labels ./labels
COPY models ./models

ENTRYPOINT ["python3", "-m", "raw_video_detector_bench.bench"]


FROM intel AS intel-npu

ARG INTEL_NPU_DRIVER_VERSION=v1.33.0
ARG INTEL_NPU_DRIVER_BUILD=20260529-26625960453

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libtbb12 \
    && mkdir -p /tmp/intel-npu \
    && cd /tmp/intel-npu \
    && curl -fsSL -o linux-npu-driver.tar.gz \
        "https://github.com/intel/linux-npu-driver/releases/download/${INTEL_NPU_DRIVER_VERSION}/linux-npu-driver-${INTEL_NPU_DRIVER_VERSION}.${INTEL_NPU_DRIVER_BUILD}-ubuntu2404.tar.gz" \
    && tar -xzf linux-npu-driver.tar.gz \
    && dpkg -i ./*.deb \
    && cd / \
    && rm -rf /tmp/intel-npu /var/lib/apt/lists/*
