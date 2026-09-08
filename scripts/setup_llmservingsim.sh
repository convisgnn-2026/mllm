#!/bin/bash
# Native (no docker, no conda) build of LLMServingSim 2.0 + ASTRA-Sim analytical backend.
# Mirrors astra-sim/Dockerfile: abseil + protobuf from source, everything under $HOME.
#   venv   : ~/venv/servingsim   (separate from the research venv; protobuf==5.29 is pinned)
#   libs   : ~/local/abseil, ~/local/protobuf
#   repo   : ~/papers/refs/code/LLMServingSim  (submodule astra-sim already checked out)
set -euo pipefail
REPO=$HOME/papers/refs/code/LLMServingSim
LOCAL=$HOME/local; SRC=$LOCAL/src; mkdir -p "$SRC"
ABSL_VER=20240722.0; PB_VER=29.0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 0) nested submodules (chakra, fmt, spdlog, network/memory backends)
cd "$REPO" && git submodule update --init --recursive --depth 1 astra-sim >/dev/null 2>&1 && log "submodules ok"

# 1) venv from the same 3.12 interpreter the research venv uses
PYBIN=$(readlink -f "$HOME/venv/jinwoo/bin/python")
[ -x "$HOME/venv/servingsim/bin/python" ] || "$PYBIN" -m venv "$HOME/venv/servingsim"
source "$HOME/venv/servingsim/bin/activate"
pip install -q -U pip
pip install -q cmake ninja numpy sympy graphviz pandas pyyaml pyinstrument rich "protobuf==5.${PB_VER}"
log "venv ready: $(python --version), cmake $(cmake --version | head -1)"

# 2) abseil
if [ ! -f "$LOCAL/abseil/lib64/cmake/absl/abslConfig.cmake" ] && [ ! -f "$LOCAL/abseil/lib/cmake/absl/abslConfig.cmake" ]; then
  cd "$SRC"; [ -d abseil-cpp-$ABSL_VER ] || { wget -q https://github.com/abseil/abseil-cpp/releases/download/$ABSL_VER/abseil-cpp-$ABSL_VER.tar.gz; tar -xf abseil-cpp-$ABSL_VER.tar.gz; }
  cmake -S abseil-cpp-$ABSL_VER -B abseil-cpp-$ABSL_VER/build -G Ninja -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LOCAL/abseil" -DABSL_PROPAGATE_CXX_STD=ON >/dev/null
  cmake --build abseil-cpp-$ABSL_VER/build --target install --parallel "$(nproc)" >/dev/null
fi
export absl_DIR=$(dirname "$(find "$LOCAL/abseil" -name abslConfig.cmake | head -1)")
log "abseil ok: $absl_DIR"

# 3) protobuf
if [ ! -x "$LOCAL/protobuf/bin/protoc" ]; then
  cd "$SRC"; [ -d protobuf-$PB_VER ] || { wget -q https://github.com/protocolbuffers/protobuf/releases/download/v$PB_VER/protobuf-$PB_VER.tar.gz; tar -xf protobuf-$PB_VER.tar.gz; }
  cmake -S protobuf-$PB_VER -B protobuf-$PB_VER/build -G Ninja -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package -DCMAKE_INSTALL_PREFIX="$LOCAL/protobuf" >/dev/null
  cmake --build protobuf-$PB_VER/build --target install --parallel "$(nproc)" >/dev/null
fi
export PATH="$LOCAL/protobuf/bin:$PATH"; export protobuf_DIR="$LOCAL/protobuf"; export PROTOBUF_FROM_SOURCE=True
export CMAKE_PREFIX_PATH="$LOCAL/protobuf:$LOCAL/abseil:${CMAKE_PREFIX_PATH:-}"
log "protobuf ok: $(protoc --version)"

# 4) chakra (python) + ASTRA-Sim analytical backend
cd "$REPO/astra-sim/extern/graph_frontend/chakra" && pip install -q . && log "chakra ok"
cd "$REPO/astra-sim" && bash ./build/astra_analytical/build.sh > "$SRC/astra_build.log" 2>&1 && log "astra-sim analytical ok" || { log "astra-sim build FAILED — see $SRC/astra_build.log"; tail -20 "$SRC/astra_build.log"; exit 1; }

# 5) env file for later shells
cat > "$HOME/venv/servingsim/env.sh" <<EOF
source $HOME/venv/servingsim/bin/activate
export PATH="$LOCAL/protobuf/bin:\$PATH"
export absl_DIR="$absl_DIR" protobuf_DIR="$LOCAL/protobuf" PROTOBUF_FROM_SOURCE=True
export CMAKE_PREFIX_PATH="$LOCAL/protobuf:$LOCAL/abseil"
EOF
log "SETUP DONE — source ~/venv/servingsim/env.sh"
