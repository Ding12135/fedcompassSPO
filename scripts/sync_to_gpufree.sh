#!/usr/bin/env bash
# 同步 fedcompass 到云服务器（远程无 rsync 时用 tar+ssh）
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gpufree4090}"
REMOTE_DIR="${REMOTE_DIR:-/root/gpufree-data/fedcompass}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "本地: $LOCAL_DIR"
echo "远程: ${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

ssh -p 31477 "root@${REMOTE_HOST#*@}" "mkdir -p ${REMOTE_DIR}" 2>/dev/null || ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

cd "${LOCAL_DIR}"
tar czf - \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='su_compass/output' \
  --exclude='examples/output' \
  . | ssh "${REMOTE_HOST}" "tar xzf - -C ${REMOTE_DIR}"

echo ""
echo "上传完成。在服务器上执行："
echo "  cd ${REMOTE_DIR}"
echo "  conda create -n fedcompass python=3.10 -y && conda activate fedcompass"
echo "  pip install -e ."
echo "  python -c \"import torch; print('cuda:', torch.cuda.is_available())\""
echo "  python -m su_compass.experiments.run_virtual_fl --algorithm fedcompass --num_global_epochs 30 --output_dir su_compass/output/smoke"
