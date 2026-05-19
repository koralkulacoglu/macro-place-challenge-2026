# Macro Placement Challenge 2026 — koral submission
#
# Pipeline: Xplace multi-seed GP (top-16 seeds) → 16 parallel SA workers
# Requires: CUDA GPU at runtime (Xplace uses GPU for routability-driven GP)
#
# ── Build (from repo root) ────────────────────────────────────────────────
#   docker build -t koral-placer -f submissions/koral/Dockerfile .
#
# ── Run a single benchmark ────────────────────────────────────────────────
#   docker run --rm --runtime=nvidia --gpus all \
#     -v $(pwd):/challenge --network none \
#     koral-placer submissions/koral/placer.py -b ibm01
#
# ── Run all 17 IBM benchmarks (full judging run) ──────────────────────────
#   docker run --rm --runtime=nvidia --gpus all \
#     -v $(pwd):/challenge --network none \
#     koral-placer submissions/koral/placer.py --all
#
# Note: Xplace requires CUDA JIT compilation on first run (~60-120s cold start).
#       Each benchmark runs for ~60 min: ~700s Xplace seed search + ~2800s parallel SA.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

SHELL ["/bin/bash", "-c"]

# ── System build dependencies ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake bison flex libfl-dev libboost-all-dev \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge

# ── Challenge framework ───────────────────────────────────────────────────
COPY macro_place /challenge/macro_place
COPY pyproject.toml requirements.txt /challenge/
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir scipy tqdm

# ── Xplace: routability-driven analytical placer ──────────────────────────
# Needs a GPU at runtime; CUDA JIT compiles kernels on first use (~60-120s).
RUN git clone --depth=1 --recurse-submodules \
        https://github.com/cuhk-eda/Xplace.git /tmp/xplace_src \
    && cd /tmp/xplace_src \
    && pip install --no-cache-dir pandas loguru seaborn matplotlib \
                                   scipy numba torchvision pulp igraph \
    && sed -i \
        's/int(torch.cuda.is_available())/int(torch.version.cuda is not None)/' \
        CMakeLists.txt \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/xplace \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=$(which python3) \
    && make -j$(nproc) \
    && make install \
    && cp -r /tmp/xplace_src/src      /opt/xplace/src \
    && cp -r /tmp/xplace_src/utils    /opt/xplace/utils \
    && cp    /tmp/xplace_src/main.py  /opt/xplace/main.py \
    && cp -r /tmp/xplace_src/cpp_to_py /opt/xplace/cpp_to_py \
    && mkdir -p /opt/xplace/thirdparty/flute \
    && cp /tmp/xplace_src/thirdparty/flute/POWV9.dat /opt/xplace/thirdparty/flute/ \
    && cp /tmp/xplace_src/thirdparty/flute/POST9.dat /opt/xplace/thirdparty/flute/ \
    && rm -rf /tmp/xplace_src

# NumPy 2.0 compatibility patches for Xplace
RUN sed -i 's/np\.round_/np.round/g'    /opt/xplace/src/core/macro_legalization.py \
    && sed -i 's/dtype=np\.bool8/dtype=np.bool_/g' /opt/xplace/src/core/macro_legalization.py

ENV XPLACE_HOME=/opt/xplace
ENV PYTHONPATH=/opt/xplace

# ── Submission code ───────────────────────────────────────────────────────
# Benchmarks are NOT baked in — provided at runtime via bind mount (-v).
# The run commands above mount the challenge repo root as /challenge,
# which gives the container access to external/MacroPlacement/Testcases/.
RUN mkdir -p /challenge/submissions/koral
COPY submissions/koral/placer.py    /challenge/submissions/koral/placer.py
COPY submissions/koral/bookshelf.py /challenge/submissions/koral/bookshelf.py
COPY submissions/koral/fast_eval.py /challenge/submissions/koral/fast_eval.py

ENTRYPOINT ["python3", "-m", "macro_place.evaluate"]
