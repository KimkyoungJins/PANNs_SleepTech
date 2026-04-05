#!/bin/bash
# ResNet22 (2.2M) — 2클래스 수면 분류

DATA_FULL="../data/data_for_ai/full_ver"
DATA_RATIO="../data/data_for_ai/ratio_ver"
WS_FULL="./workspaces/full_ver"
WS_RATIO="./workspaces/ratio_ver"
PRETRAINED="./ResNet22_mAP=0.430.pth"

echo "=============================================="
echo "  [1/4] full_ver (ResNet22 + Focal + Oversample)"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_FULL --workspace=$WS_FULL \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=focal --oversample --cuda

echo "=============================================="
echo "  [2/4] ratio_ver (ResNet22 + 기본 CE)"
echo "=============================================="
python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO --workspace=$WS_RATIO \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --cuda

echo "=============================================="
echo "  [3/4] 테스트"
echo "=============================================="
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=$WS_FULL --batch_size=16 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=$WS_RATIO --batch_size=16 --cuda

echo "=============================================="
echo "  [4/4] 비교"
echo "=============================================="
python3 pytorch/main.py compare --workspace_full=$WS_FULL --workspace_ratio=$WS_RATIO

echo "완료!"
