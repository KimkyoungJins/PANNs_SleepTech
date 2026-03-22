#!/bin/bash
# ===== 데이터 추가 후 이어서 학습 =====
#
# 새 환자 데이터가 추가되었을 때:
#   1. process_sleep_data.py 다시 실행 → full_ver, ratio_ver 재생성
#   2. 이 스크립트 실행 → 기존 best_model.pth에서 이어서 학습
#
# 사용법:
#   bash scripts/6_resume_train.sh

DATA_FULL="../data_for_ai/full_ver"
DATA_RATIO="../data_for_ai/ratio_ver"
WS_FULL="./workspaces/full_ver"
WS_RATIO="./workspaces/ratio_ver"

echo "=============================================="
echo "  full_ver 이어서 학습 (기존 모델에서 추가 학습)"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_FULL \
    --workspace=$WS_FULL \
    --resume_path=$WS_FULL/checkpoints/best_model.pth \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --num_epochs=80 \
    --cuda

echo "=============================================="
echo "  ratio_ver 이어서 학습"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO \
    --workspace=$WS_RATIO \
    --resume_path=$WS_RATIO/checkpoints/best_model.pth \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --num_epochs=80 \
    --cuda

echo "=============================================="
echo "  테스트 + 비교"
echo "=============================================="
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=$WS_FULL --batch_size=16 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=$WS_RATIO --batch_size=16 --cuda
python3 pytorch/main.py compare --workspace_full=$WS_FULL --workspace_ratio=$WS_RATIO

echo ""
echo "완료!"
