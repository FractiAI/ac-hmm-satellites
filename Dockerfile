FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ncbi-blast+ \
    python3-dev \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements_lock.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements_lock.txt

COPY . /workspace/
ENV PYTHONPATH=/workspace/src/python

RUN pip3 install pybind11 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc) && \
    cp achmm_trellis*.so /workspace/src/python/ 2>/dev/null || true

CMD ["bash"]
