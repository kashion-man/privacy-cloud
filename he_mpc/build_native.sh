#!/usr/bin/env bash
# WHU216: official defaults for real SoftSpokenOT, compiled without ADX.
set -euo pipefail
HE_MPC_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
HE_MPC_BUILD_ENV="$HE_MPC_ROOT/.build-env"
HE_MPC_SOURCE="$HE_MPC_ROOT/third_party/mp-spdz-0.4.4"
HE_MPC_CONDA="${HE_MPC_CONDA:-conda}"
if ! test -x "$HE_MPC_BUILD_ENV/bin/x86_64-conda-linux-gnu-g++"; then
  "$HE_MPC_CONDA" create -y -p "$HE_MPC_BUILD_ENV" --override-channels -c conda-forge \
    gxx_linux-64=12 make cmake gmp libsodium boost-cpp openssl pkg-config
fi
export PATH="$HE_MPC_BUILD_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$HE_MPC_BUILD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$HE_MPC_SOURCE"
cat > CONFIG.mine <<EOF
CXX = $HE_MPC_BUILD_ENV/bin/x86_64-conda-linux-gnu-g++
ARCH = -march=haswell
USE_KOS = 0
USE_LIBOTE2 = 1
AVX_OT = 0
AVX_SIMPLEOT = 0
GDEBUG =
ERROR =
MY_CFLAGS = -I$HE_MPC_BUILD_ENV/include
MY_LDLIBS = -L$HE_MPC_BUILD_ENV/lib -Wl,-rpath,$HE_MPC_BUILD_ENV/lib
OTE_OPTS = --par=4 -DOC_THIRDPARTY_CLONE_DIR=$HE_MPC_SOURCE/deps/ot-fetch -DCOPROTO_GIT_TAG=hempc-v044-coproto -DMACORO_GIT_TAG=hempc-v044-macoro -DFETCH_BOOST=OFF -DPARALLEL_FETCH=4
EOF
function ensure_source() {
  local source_name="$1" source_url="$2" source_pin="$3" source_tag="${4:-}"
  mkdir -p "deps/$source_name"
  if ! test -e "deps/$source_name/.git"; then
    git -C "deps/$source_name" init
    git -C "deps/$source_name" remote add origin "$source_url"
  fi
  if test "$(git -C "deps/$source_name" rev-parse HEAD 2>/dev/null || true)" != "$source_pin"; then
    git -C "deps/$source_name" -c http.version=HTTP/1.1 fetch --depth 1 origin "$source_pin"
    git -C "deps/$source_name" checkout --detach FETCH_HEAD
  fi
  if test -n "$source_tag"; then git -C "deps/$source_name" tag -f "$source_tag" HEAD; fi
}
ensure_source ot-fetch/coproto https://github.com/mkskeller/coproto 8dce3918b06c80dbcd65baeb899f1a694433f926 hempc-v044-coproto
ensure_source ot-fetch/macoro https://github.com/mkskeller/macoro ee5fa61a9bc2e4cd4cdddea8d4a6485332e8d58e hempc-v044-macoro
ensure_source ot-fetch/libdivide https://github.com/ridiculousfish/libdivide 66190e9daa603cabe95d99f09fb79b5c186d0417 v5.2.0
if ! test -d deps/libOTe2/.git; then
  git -C deps/libOTe2 init
  git -C deps/libOTe2 remote add origin https://github.com/mkskeller/libOTe
fi
if test "$(git -C deps/libOTe2 rev-parse HEAD 2>/dev/null || true)" != fe6b2ff82a976a5332b3a2efe325831588b1324a; then
  git -C deps/libOTe2 -c http.version=HTTP/1.1 fetch --depth 1 origin fe6b2ff82a976a5332b3a2efe325831588b1324a
fi
git -C deps/libOTe2 checkout fe6b2ff82a976a5332b3a2efe325831588b1324a
git -C deps/libOTe2 -c http.version=HTTP/1.1 submodule update --init --recursive --depth 1
make -j"${HE_MPC_BUILD_JOBS:-4}" semi-party.x
"$HE_MPC_CONDA" list -p "$HE_MPC_BUILD_ENV" --explicit > "$HE_MPC_ROOT/build-environment.txt"
sha256sum semi-party.x libSPDZ.so > "$HE_MPC_ROOT/native-runtime.sha256"
