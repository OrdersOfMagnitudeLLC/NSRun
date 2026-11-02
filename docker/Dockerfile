# NSRun KVBox serverless image — llama-server with --kv-box by default.
# Build:  docker build -f docker/Dockerfile -t ghcr.io/fomesdise/nsrun:latest .
# Run:    docker run --gpus all -p 8080:8080 -v nsrun-models:/models ghcr.io/fomesdise/nsrun:latest

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*
# CMake >= 3.30 required: the build sets CXX/CUDA dialect 20.
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.tar.gz \
        | tar xz -C /opt \
    && ln -sf /opt/cmake-3.31.5-linux-x86_64/bin/cmake /usr/local/bin/cmake
WORKDIR /src
COPY . .
RUN cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89" \
 && cmake --build build -j"$(nproc)" --target llama-server

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_PATH=/models/Qwen3-14B-Q4_K_M.gguf \
    MODEL_URL=https://modelscope.cn/models/Qwen/Qwen3-14B-GGUF/resolve/master/Qwen3-14B-Q4_K_M.gguf \
    EXPECTED_SHA256=500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0 \
    CTX=3145728 \
    PARALLEL=3 \
    PORT=8080

VOLUME /models
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
