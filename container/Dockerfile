# Ants-base - Obtain ANTs binaries, pre-built and optimized ANTs image.
FROM antsx/ants:latest as ants-base

# Build the final image
FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies necessary for running ANTs and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    # X and ANTs dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    libstdc++6 \
    # Crucial OpenMP and BLAS/ATLAS libraries for ANTs performance and execution
    libgomp1 \
    libatlas3-base \
    ca-certificates \
    # Add C Runtime libraries and debugging utilities
    locales \
    libfontconfig1 \
    patchelf \ 
    && rm -rf /var/lib/apt/lists/*

# Configure the localization environment
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Copy ALL ANTs binaries from the first stage
COPY --from=ants-base /usr/bin/ /usr/local/bin/
COPY --from=ants-base /usr/local/bin/ /usr/local/bin/
COPY --from=ants-base /opt/ants/bin/ /usr/local/bin/

COPY --from=ants-base /opt/ants/lib/ /usr/local/lib/ants/
ENV LD_LIBRARY_PATH="/usr/local/lib/ants/:$LD_LIBRARY_PATH"

RUN chmod +x /usr/local/bin/* 2>/dev/null || true

RUN echo "=== Binarios de ANTs copiados ===" \
    && ls -l /usr/local/bin/antsRegistrationSyN.sh 2>/dev/null || echo "Buscando alternativamente..." \
    && find /usr/local/bin -name "*ants*" -type f | head -10

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY pipeline.py /workspace/pipeline.py

COPY src /workspace/src
RUN pip install -e /workspace/src

COPY run_pipeline.sh /workspace/run_pipeline.sh
RUN chmod +x /workspace/run_pipeline.sh

ENTRYPOINT ["/workspace/run_pipeline.sh"]