#!/bin/bash
# ============================================================
# 전체 학습 파이프라인 v2
# ResNet22 (4가지) + EOG_CNN + Hierarchical 평가
#
# 실행: cd project1/resnet && bash scripts/9_train_all_v2.sh
# ============================================================

set -e  # 에러 시 중단

# ─── 경로 설정 ───
DATA_FULL_3CLASS="../data/data_for_ai/full_ver_3class"
DATA_FULL_2CLASS="../data/data_for_ai/full_ver_2class"
DATA_RATIO_3CLASS="../data/data_for_ai/ratio_ver_3class"
DATA_RATIO_2CLASS="../data/data_for_ai/ratio_ver_2class"
DATA_REM_NREM="../data/data_for_ai/full_ver_rem_nrem"
PRETRAINED="./ResNet22_mAP=0.430.pth"
UTILS="./utils"
HIER_VERSION="v2"

echo ""
echo "============================================================"
echo "  전체 학습 파이프라인 시작"
echo "  데이터: $DATA_FULL_3CLASS"
echo "  Hierarchical 버전: $HIER_VERSION"
echo "============================================================"

# ─── Step 0: REM/NREM CSV 생성 ───
echo ""
echo "####################################################"
echo "  Step 0: REM/NREM CSV 생성"
echo "####################################################"
cd ..
python3 make_rem_nrem_csv.py
cd resnet

# ============================================================
#  PART 1: ResNet22 2클래스 (Wake / Sleep)
# ============================================================
echo ""
echo "####################################################"
echo "  PART 1: ResNet22 2클래스 (Wake / Sleep)"
echo "####################################################"

cp $UTILS/config_2class.py $UTILS/config.py
cp $UTILS/data_generator.py $UTILS/data_generator_backup.py 2>/dev/null

# 1-1. full_ver_2class 학습
echo ""
echo ">>> [1/4] full_ver_2class 학습..."
python3 pytorch/main.py train \
    --data_dir=$DATA_FULL_2CLASS \
    --workspace=./workspaces/full_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

# 1-2. ratio_ver_2class 학습
echo ""
echo ">>> [2/4] ratio_ver_2class 학습..."
python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO_2CLASS \
    --workspace=./workspaces/ratio_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --patience=10 --cuda

# 1-3. 2클래스 테스트
echo ""
echo ">>> 2클래스 테스트..."
python3 pytorch/main.py test --data_dir=$DATA_FULL_2CLASS --workspace=./workspaces/full_ver_2class --batch_size=8 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL_2CLASS --workspace=./workspaces/ratio_ver_2class --batch_size=8 --cuda
python3 pytorch/main.py compare --workspace_full=./workspaces/full_ver_2class --workspace_ratio=./workspaces/ratio_ver_2class

# ============================================================
#  PART 2: ResNet22 3클래스 (Wake / REM / NREM)
# ============================================================
echo ""
echo "####################################################"
echo "  PART 2: ResNet22 3클래스 (Wake / REM / NREM)"
echo "####################################################"

cp $UTILS/config_3class.py $UTILS/config.py
cp $UTILS/data_generator_3class.py $UTILS/data_generator.py

# 2-1. full_ver_3class 학습
echo ""
echo ">>> [3/4] full_ver_3class 학습..."
python3 pytorch/main.py train \
    --data_dir=$DATA_FULL_3CLASS \
    --workspace=./workspaces/full_ver_3class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

# 2-2. ratio_ver_3class 학습
echo ""
echo ">>> [4/4] ratio_ver_3class 학습..."
python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO_3CLASS \
    --workspace=./workspaces/ratio_ver_3class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --patience=10 --cuda

# 2-3. 3클래스 테스트
echo ""
echo ">>> 3클래스 테스트..."
python3 pytorch/main.py test --data_dir=$DATA_FULL_3CLASS --workspace=./workspaces/full_ver_3class --batch_size=8 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL_3CLASS --workspace=./workspaces/ratio_ver_3class --batch_size=8 --cuda
python3 pytorch/main.py compare --workspace_full=./workspaces/full_ver_3class --workspace_ratio=./workspaces/ratio_ver_3class

# config 복원
cp $UTILS/config_2class.py $UTILS/config.py
cp $UTILS/data_generator_backup.py $UTILS/data_generator.py 2>/dev/null

# ============================================================
#  PART 3: EOG_CNN (REM / NREM)
# ============================================================
echo ""
echo "####################################################"
echo "  PART 3: EOG_CNN 1D CNN (REM / NREM)"
echo "####################################################"

# 3-1. EOG_CNN 학습
echo ""
echo ">>> EOG_CNN 학습..."
python3 train_eog_cnn.py --version $HIER_VERSION \
    --data_dir=$DATA_FULL_3CLASS \
    --csv_dir=$DATA_REM_NREM \
    --epochs=50 --batch_size=128 \
    --lr=1e-3 --weight_decay=1e-4 \
    --focal_alpha=0.75 --focal_gamma=3.0 \
    --patience=15

# 3-2. EOG_CNN 테스트
echo ""
echo ">>> EOG_CNN 테스트..."
python3 eval_eog_cnn.py --version $HIER_VERSION \
    --data_dir=$DATA_FULL_3CLASS \
    --csv_dir=$DATA_REM_NREM

# ============================================================
#  PART 4: Hierarchical 3-Class 평가
# ============================================================
echo ""
echo "####################################################"
echo "  PART 4: Hierarchical 3-Class 평가"
echo "####################################################"

# 4-1. 최고 ResNet22 2-class 모델 선택 (full vs ratio 비교)
echo ""
echo ">>> 최고 ResNet22 2-class 모델 선택..."
python3 -c "
import torch, json

candidates = {
    'full_ver_2class': './workspaces/full_ver_2class/checkpoints/best_model.pth',
    'ratio_ver_2class': './workspaces/ratio_ver_2class/checkpoints/best_model.pth',
}

best_name = None
best_acc = 0
for name, path in candidates.items():
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    acc = ckpt.get('val_acc', 0)
    print(f'  {name}: val_acc={acc:.4f}, epoch={ckpt.get(\"epoch\", \"?\")}')
    if acc > best_acc:
        best_acc = acc
        best_name = name

print(f'\n  최고: {best_name} (val_acc={best_acc:.4f})')

# best를 hierarchical 버전 폴더에 복사
import shutil, os
ver_dir = f'./checkpoints/hierarchical/$HIER_VERSION'
os.makedirs(ver_dir, exist_ok=True)
src = candidates[best_name]
dst = os.path.join(ver_dir, 'stage1_resnet22.pth')
shutil.copy2(src, dst)
print(f'  복사: {src} → {dst}')

# 선택 정보 저장
with open(os.path.join(ver_dir, 'stage1_source.txt'), 'w') as f:
    f.write(f'{best_name}\nval_acc={best_acc:.4f}\n')
"

# 4-2. Hierarchical 추론 (balanced + full)
echo ""
echo ">>> Hierarchical 추론..."
python3 hierarchical_infer.py --version $HIER_VERSION \
    --data_dir=$DATA_FULL_3CLASS \
    --test_type=both \
    --test_csv_balanced=${DATA_FULL_3CLASS}/test.csv \
    --test_csv_full=${DATA_FULL_3CLASS}/test_full.csv

# ============================================================
#  완료
# ============================================================
echo ""
echo "============================================================"
echo "  전체 학습 파이프라인 완료!"
echo "============================================================"
echo ""
echo "ResNet22 결과:"
echo "  2클래스 full:  ./workspaces/full_ver_2class/results/"
echo "  2클래스 ratio: ./workspaces/ratio_ver_2class/results/"
echo "  3클래스 full:  ./workspaces/full_ver_3class/results/"
echo "  3클래스 ratio: ./workspaces/ratio_ver_3class/results/"
echo ""
echo "EOG_CNN 결과:"
echo "  ./checkpoints/hierarchical/$HIER_VERSION/results/eog_cnn_test.json"
echo ""
echo "Hierarchical 3-Class 최종 결과:"
echo "  ./checkpoints/hierarchical/$HIER_VERSION/results/test_balanced.json"
echo "  ./checkpoints/hierarchical/$HIER_VERSION/results/test_full.json"
echo "  ./checkpoints/hierarchical/$HIER_VERSION/config.json"
echo ""

# 이메일 알림
python3 send_email.py "전체 학습 완료 ($HIER_VERSION)" \
    "ResNet22 (4가지) + EOG_CNN + Hierarchical 평가 완료" 2>/dev/null || true
