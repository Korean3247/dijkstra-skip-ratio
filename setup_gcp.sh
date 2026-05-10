#!/usr/bin/env bash
# =============================================================
# GCP VM 생성 + 실험 환경 설정 스크립트
# 사전 조건: gcloud CLI 설치 및 `gcloud auth login` 완료
# =============================================================

set -euo pipefail

# ── 설정값 (필요 시 수정) ────────────────────────────────────
PROJECT_ID="$(gcloud config get-value project)"   # 현재 프로젝트 자동 감지
INSTANCE_NAME="dijkstra-bench"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-4"     # 4 vCPU, 16 GB RAM
DISK_SIZE="50GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
# ────────────────────────────────────────────────────────────

echo "================================================================"
echo " Project  : $PROJECT_ID"
echo " Instance : $INSTANCE_NAME  ($MACHINE_TYPE)"
echo " Zone     : $ZONE"
echo "================================================================"

# 1. VM 생성
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type="pd-ssd" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --metadata=startup-script='#!/bin/bash
    apt-get update -y
    apt-get install -y python3-pip python3-dev git
    pip3 install --upgrade pip
    pip3 install numpy matplotlib networkx scipy pandas tqdm
    echo "Python packages installed." >> /tmp/setup_done.txt
  '

echo ""
echo "VM 생성 완료. 30초 후 SSH 연결 시도..."
sleep 30

# 2. 실험 파일 업로드
echo "파일 업로드 중..."
gcloud compute scp \
  dijkstra_benchmark.py \
  requirements.txt \
  run_experiment.sh \
  "$INSTANCE_NAME":~ \
  --zone="$ZONE"

# 3. 실험 실행 (nohup으로 백그라운드)
echo ""
echo "실험 시작 (백그라운드)..."
gcloud compute ssh "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --command="nohup bash ~/run_experiment.sh > ~/bench.log 2>&1 &"

echo ""
echo "================================================================"
echo " 실험이 백그라운드에서 실행 중입니다."
echo ""
echo " 로그 확인:    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f ~/bench.log'"
echo " 결과 다운로드: gcloud compute scp --recurse $INSTANCE_NAME:~/results ./ --zone=$ZONE"
echo " VM 삭제:      gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo "================================================================"
