# honion needs no CUDA toolkit at build time — the kernel is compiled at run
# time via NVRTC — so the build stage is a stock Rust image and only the
# runtime stage needs libnvrtc (the NVIDIA driver, libcuda.so.1, is injected
# by the host's container toolkit, never baked into the image).
#
#   docker build -t honion .
#   docker run --rm --gpus all -v "$PWD/keys:/keys" honion \
#       search --prefix carroll --out /keys --count 1
#
# The RUNTIME_IMAGE must provide libnvrtc 13.x (the binary dlopens
# libnvrtc.so.13, matching the bindings in crates/honion-gpu); adjust the
# tag to whatever -nvrtc image your registry actually has.

ARG BUILDER=rust:1.85.1-trixie
ARG RUNTIME_IMAGE=nvidia/cuda:13.1.0-nvrtc-ubuntu24.04

FROM ${BUILDER} AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY cuda ./cuda
RUN cargo build --release -p honion-cli

FROM ${RUNTIME_IMAGE}
COPY --from=build /src/target/release/honion /usr/local/bin/honion
ENTRYPOINT ["honion"]
