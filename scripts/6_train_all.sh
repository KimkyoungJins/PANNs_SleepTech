#!/bin/bash
# Cnn6 — 2클래스 + 3클래스 전체 실험

DATA_FULL="../data_for_ai/full_ver"
DATA_RATIO="../data_for_ai/ratio_ver"
PRETRAINED="./Cnn6_mAP=0.343.pth"
UTILS="./utils"

# ======================================================
#  PART 1: 2클래스 (sleep / wake)
# ======================================================
echo ""
echo "####################################################"
echo "  PART 1: 2클래스 (sleep / wake)"
echo "####################################################"

# config 교체: 2클래스
cp $UTILS/config_2class.py $UTILS/config.py
cp $UTILS/data_generator.py $UTILS/data_generator_backup.py 2>/dev/null

python3 pytorch/main.py train \
    --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --cuda

python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO --workspace=./workspaces/ratio_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --cuda

python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_2class --batch_size=16 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/ratio_ver_2class --batch_size=16 --cuda
python3 pytorch/main.py compare --workspace_full=./workspaces/full_ver_2class --workspace_ratio=./workspaces/ratio_ver_2class

# ======================================================
#  PART 2: 3클래스 (rem / nrem / wake)
# ======================================================
echo ""
echo "####################################################"
echo "  PART 2: 3클래스 (rem / nrem / wake)"
echo "####################################################"

# config 교체: 3클래스
cp $UTILS/config_3class.py $UTILS/config.py
cp $UTILS/data_generator_3class.py $UTILS/data_generator.py

python3 pytorch/main.py train \
    --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_3class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --cuda

python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO --workspace=./workspaces/ratio_ver_3class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=16 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --cuda

python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_3class --batch_size=16 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/ratio_ver_3class --batch_size=16 --cuda
python3 pytorch/main.py compare --workspace_full=./workspaces/full_ver_3class --workspace_ratio=./workspaces/ratio_ver_3class

# 원래 config 복원 (2클래스 기본)
cp $UTILS/config_2class.py $UTILS/config.py
cp $UTILS/data_generator_backup.py $UTILS/data_generator.py 2>/dev/null

echo ""
echo "####################################################"
echo "  전체 완료!"
echo "####################################################"
echo ""
echo "결과 위치:"
echo "  2클래스 full:  ./workspaces/full_ver_2class/results/report.png"
echo "  2클래스 ratio: ./workspaces/ratio_ver_2class/results/report.png"
echo "  2클래스 비교:  ./workspaces/comparison/comparison.png"
echo "  3클래스 full:  ./workspaces/full_ver_3class/results/report.png"
echo "  3클래스 ratio: ./workspaces/ratio_ver_3class/results/report.png"
echo "  3클래스 비교:  ./workspaces/comparison/comparison.png"
