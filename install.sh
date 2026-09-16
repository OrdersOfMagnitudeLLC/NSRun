#!/bin/bash
set -e
echo "Building NSRun..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNS_ENABLE_KVBOX=ON
cmake --build build --config Release -j$(nproc)
echo "Creating nsrun symlink..."
ln -sf build/bin/llama-cli nsrun
echo "Done. Run: ./nsrun --help"
