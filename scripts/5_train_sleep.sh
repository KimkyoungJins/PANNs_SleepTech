#!/bin/bash
# ===== 수면 단계 분류 — 전체 학습 파이프라인 =====
#
# 모델 A (full_ver): Focal Loss + Oversampling + Weighted CE 적용
# 모델 B (ratio_ver): 기본 CE, 밸런싱 데이터 그대로
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/5_train_sleep.sh

# ─── 경로 설정 ───
DATA_FULL="../data_for_ai/full_ver"
DATA_RATIO="../data_for_ai/ratio_ver"
WS_FULL="./workspaces/full_ver"
WS_RATIO="./workspaces/ratio_ver"
PRETRAINED="./Cnn14_16k_mAP=0.438.pth"

echo "=============================================="
echo "  [1/4] full_ver 학습 (개선: Focal + Oversample)"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_FULL \
    --workspace=$WS_FULL \
    --pretrained_path=$PRETRAINED \
    --freeze_cnn \
    --batch_size=16 \
    --learning_rate=1e-4 \
    --num_epochs=50 \
    --loss_type=clip_ce \
    --oversample \
    --cuda

echo "=============================================="
echo "  [2/4] ratio_ver 학습 (기본 CE, 밸런싱 데이터)"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO \
    --workspace=$WS_RATIO \
    --pretrained_path=$PRETRAINED \
    --freeze_cnn \
    --batch_size=16 \
    --learning_rate=1e-4 \
    --num_epochs=50 \
    --loss_type=clip_ce \
    --cuda

echo "=============================================="
echo "  [3/4] 테스트 평가 + 리포트 생성"
echo "=============================================="
python3 pytorch/main.py test \
    --data_dir=$DATA_FULL \
    --workspace=$WS_FULL \
    --batch_size=16 \
    --cuda

python3 pytorch/main.py test \
    --data_dir=$DATA_FULL \
    --workspace=$WS_RATIO \
    --batch_size=16 \
    --cuda

echo "=============================================="
echo "  [4/4] 두 모델 비교"
echo "=============================================="
python3 pytorch/main.py compare \
    --workspace_full=$WS_FULL \
    --workspace_ratio=$WS_RATIO

echo ""
echo "완료! 결과 확인:"
echo "  full_ver  리포트: $WS_FULL/results/report.png"
echo "  ratio_ver 리포트: $WS_RATIO/results/report.png"
echo "  비교 리포트:      ./workspaces/comparison/comparison.png"
