#!/bin/bash
set -e
echo "[TinyTensor] Building the library..."
cd tinytensor
mkdir -p build
cd build
cmake .. -Wno-dev
make -j$(nproc)
cd ../..
rm -rf tinytensor/build
echo "[TinyTensor] Finished."
