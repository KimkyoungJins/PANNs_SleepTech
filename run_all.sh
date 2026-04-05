#!/bin/bash
# CNN6 전체 실험 자동 실행 (4개: 2class/3class × full/ratio)

BASE=/home/yk/Desktop/sleeptech/project1
CNN6=$BASE/cnn6
DATA=$BASE/data/data_for_ai
PRETRAINED=$CNN6/Cnn6_mAP=0.343.pth

echo "============================================================"
echo "CNN6 전체 실험 시작"
echo "============================================================"

# ===== 실험 1: full_ver 2-Class =====
echo ""
echo "============================================================"
echo "[1/4] full_ver 2-Class"
echo "============================================================"
cp $CNN6/utils/config_2class.py $CNN6/utils/config.py
cd $CNN6/pytorch
python3 main.py train \
  --data_dir=$DATA/full_ver_2class \
  --workspace=$CNN6/workspaces/full_ver_2class \
  --pretrained_path=$PRETRAINED \
  --freeze_cnn --batch_size=16 --num_epochs=50 --cuda

python3 main.py test \
  --data_dir=$DATA/full_ver_2class \
  --workspace=$CNN6/workspaces/full_ver_2class \
  --cuda

# ===== 실험 2: ratio_ver 2-Class =====
echo ""
echo "============================================================"
echo "[2/4] ratio_ver 2-Class"
echo "============================================================"
cp $CNN6/utils/config_2class.py $CNN6/utils/config.py
cd $CNN6/pytorch
python3 main.py train \
  --data_dir=$DATA/ratio_ver_2class \
  --workspace=$CNN6/workspaces/ratio_ver_2class \
  --pretrained_path=$PRETRAINED \
  --freeze_cnn --batch_size=16 --num_epochs=50 --cuda

python3 main.py test \
  --data_dir=$DATA/ratio_ver_2class \
  --workspace=$CNN6/workspaces/ratio_ver_2class \
  --cuda

# ===== 실험 3: full_ver 3-Class =====
echo ""
echo "============================================================"
echo "[3/4] full_ver 3-Class"
echo "============================================================"
cp $CNN6/utils/config_3class.py $CNN6/utils/config.py
cd $CNN6/pytorch
python3 main.py train \
  --data_dir=$DATA/full_ver \
  --workspace=$CNN6/workspaces/full_ver_3class \
  --pretrained_path=$PRETRAINED \
  --freeze_cnn --batch_size=16 --num_epochs=50 --cuda

python3 main.py test \
  --data_dir=$DATA/full_ver \
  --workspace=$CNN6/workspaces/full_ver_3class \
  --cuda

# ===== 실험 4: ratio_ver 3-Class =====
echo ""
echo "============================================================"
echo "[4/4] ratio_ver 3-Class"
echo "============================================================"
cp $CNN6/utils/config_3class.py $CNN6/utils/config.py
cd $CNN6/pytorch
python3 main.py train \
  --data_dir=$DATA/ratio_ver \
  --workspace=$CNN6/workspaces/ratio_ver_3class \
  --pretrained_path=$PRETRAINED \
  --freeze_cnn --batch_size=16 --num_epochs=50 --cuda

python3 main.py test \
  --data_dir=$DATA/ratio_ver \
  --workspace=$CNN6/workspaces/ratio_ver_3class \
  --cuda

echo ""
echo "============================================================"
echo "CNN6 전체 실험 완료!"
echo "============================================================"
