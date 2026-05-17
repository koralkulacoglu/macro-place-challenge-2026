# Macro Placement Challenge — koral submission
#
# Build (from repo root):
#   docker build -t koral-placer -f submissions/koral/Dockerfile .
#
# Dev loop (bind-mount repo, GPU passthrough):
#   docker run --rm --runtime=nvidia --gpus all -v $(pwd):/challenge \
#     --entrypoint python koral-placer \
#     -m macro_place.evaluate submissions/koral/placer.py -b ibm01

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake bison flex libfl-dev libboost-all-dev libcairo2-dev \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge

# Use bash for all subsequent RUN commands (dash default doesn't support pipefail)
SHELL ["/bin/bash", "-c"]

# Challenge framework
COPY macro_place /challenge/macro_place
COPY pyproject.toml requirements.txt /challenge/
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir scipy tqdm

# Python deps DREAMPlace needs at runtime
RUN pip install --no-cache-dir \
        pyunpack patool cairocffi pkgconfig shapely \
        torch_optimizer ncg_optimizer

# Clone DREAMPlace (network is available at build time; --network none is runtime-only)
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
RUN git clone --depth=1 --recursive https://github.com/limbo018/DREAMPlace.git /tmp/dreamplace_src

# Apply compatibility patches (CUDA 12.4 / NumPy 2.0 / detailed-placement disabled)
COPY submissions/koral/patch_dreamplace.sh /tmp/patch_dreamplace.sh
RUN bash /tmp/patch_dreamplace.sh /tmp/dreamplace_src

# Make libcuda.so stub visible so cmake's find_package(CUDA) succeeds without a GPU device
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so

# Build and install DREAMPlace into /opt so bind-mounts on /challenge don't shadow it
RUN cd /tmp/dreamplace_src \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/dreamplace \
        -DCUDA_ARCHITECTURES="8.0;8.6;8.9;9.0" \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/dreamplace_src /tmp/patch_dreamplace.sh

# DREAMPlace at /opt so bind-mounts on /challenge don't shadow it
ENV PYTHONPATH=/opt/dreamplace:/opt/dreamplace/dreamplace

# Benchmarks (baked in for self-contained image; judges mount their own)
COPY external/MacroPlacement/Testcases/ICCAD04 /challenge/external/MacroPlacement/Testcases/ICCAD04
COPY external/MacroPlacement/CodeElements/Plc_client /challenge/external/MacroPlacement/CodeElements/Plc_client

# Our submission code
RUN mkdir -p /challenge/submissions/koral
COPY submissions/koral/placer.py    /challenge/submissions/koral/placer.py
COPY submissions/koral/bookshelf.py /challenge/submissions/koral/bookshelf.py

ENTRYPOINT ["python", "-m", "macro_place.evaluate"]
