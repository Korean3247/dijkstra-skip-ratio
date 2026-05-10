#!/usr/bin/env bash
# VM 내부에서 실행되는 실험 스크립트
set -euo pipefail

echo "=== Dijkstra Benchmark Start: $(date) ==="

# 패키지 확인 후 설치
pip3 install -q -r ~/requirements.txt 2>/dev/null || true

cd ~
mkdir -p results

# 실험 실행
python3 dijkstra_benchmark.py

echo ""
echo "=== Done: $(date) ==="
echo "Results:"
ls -lh ~/results/
