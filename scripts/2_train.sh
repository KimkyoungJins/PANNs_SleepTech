#!/bin/bash
# ============================================================
# BiLSTM REM/NREM 학습 (v1)
#
# 선행조건: 1_extract_embeddings.sh 가 완료되어
#         bilstm/embeddings/ 에 .npy 파일들이 있어야 함.
#
# 실행: cd project1/bilstm && bash scripts/2_train.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

WORKSPACE="./workspaces/v1"

echo ""
echo "============================================================"
echo "  BiLSTM REM/NREM 학습 (v1)"
echo "  Workspace: $WORKSPACE"
echo "  Embeddings: ./embeddings/"
echo "============================================================"
echo ""

python3 train.py \
    --embeddings_dir=./embeddings \
    --workspace=$WORKSPACE \
    --seq_len=40 --center_window=20 --stride=20 \
    --batch_size=32 --num_epochs=50 \
    --learning_rate=1e-3 --weight_decay=1e-4 \
    --patience=15 --lr_patience=5 --grad_clip=1.0 \
    --focal_alpha=0.75 --focal_gamma=3.0 \
    --proj_dim=256 --hidden_dim=128 --num_layers=2 --dropout=0.3 \
    --num_workers=2 --seed=42 \
    --cuda
