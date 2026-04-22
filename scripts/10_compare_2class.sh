#!/bin/bash
# ============================================================
# CNN6 vs ResNet22 — 2-Class 공정 비교 학습
# 동일 조건: full_ver_2class, clip_ce, oversample, batch=8, patience=10
#
# 실행: cd project1/resnet && bash scripts/10_compare_2class.sh
# ============================================================

set -e

DATA="../data/data_for_ai/full_ver_2class"
UTILS="./utils"

echo ""
echo "============================================================"
echo "  CNN6 vs ResNet22 — 2-Class 공정 비교"
echo "  데이터: $DATA (149명, full_ver)"
echo "  조건: clip_ce, oversample, batch=8, patience=10"
echo "============================================================"

# ── config 설정 (2-class) ──
cp $UTILS/config_2class.py $UTILS/config.py

# ============================================================
#  1. ResNet22 학습 + 테스트
# ============================================================
echo ""
echo "####################################################"
echo "  [1/2] ResNet22 2-class 학습"
echo "####################################################"

python3 pytorch/main.py train \
    --data_dir=$DATA \
    --workspace=./workspaces/full_ver_2class \
    --pretrained_path=./ResNet22_mAP=0.430.pth --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

echo ""
echo ">>> ResNet22 테스트..."
python3 pytorch/main.py test \
    --data_dir=$DATA \
    --workspace=./workspaces/full_ver_2class \
    --batch_size=8 --cuda

# ============================================================
#  2. CNN6 학습 + 테스트
# ============================================================
echo ""
echo "####################################################"
echo "  [2/2] CNN6 2-class 학습"
echo "####################################################"

# CNN6 config도 2-class로 설정
cp ../cnn6/utils/config_2class.py ../cnn6/utils/config.py

python3 ../cnn6/pytorch/main.py train \
    --data_dir=$DATA \
    --workspace=../cnn6/workspaces/full_ver_2class \
    --pretrained_path=../cnn6/Cnn6_mAP=0.343.pth --freeze_cnn \
    --batch_size=8 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

echo ""
echo ">>> CNN6 테스트..."
python3 ../cnn6/pytorch/main.py test \
    --data_dir=$DATA \
    --workspace=../cnn6/workspaces/full_ver_2class \
    --batch_size=8 --cuda

# ============================================================
#  3. 결과 비교
# ============================================================
echo ""
echo "####################################################"
echo "  결과 비교"
echo "####################################################"

python3 -c "
import json

models = {
    'ResNet22': './workspaces/full_ver_2class/results/test_results.json',
    'CNN6': '../cnn6/workspaces/full_ver_2class/results/test_results.json',
}

print()
print('=' * 65)
print('  CNN6 vs ResNet22 — 2-Class 공정 비교 결과 (149명 데이터)')
print('=' * 65)
print(f'{\"\":>10s} {\"Accuracy\":>10s} {\"Wake P\":>8s} {\"Wake R\":>8s} {\"Wake F1\":>8s} {\"Sleep P\":>8s} {\"Sleep R\":>8s} {\"Sleep F1\":>8s}')
print('-' * 65)

for name, path in models.items():
    with open(path) as f:
        r = json.load(f)
    w = r['per_class']['wake']
    s = r['per_class']['sleep']
    print(f'{name:>10s} {r[\"accuracy\"]*100:>9.2f}% {w[\"precision\"]:>8.4f} {w[\"recall\"]:>8.4f} {w[\"f1\"]:>8.4f} {s[\"precision\"]:>8.4f} {s[\"recall\"]:>8.4f} {s[\"f1\"]:>8.4f}')

print()
"

echo ""
echo "============================================================"
echo "  완료!"
echo "============================================================"
echo ""
echo "결과:"
echo "  ResNet22: ./workspaces/full_ver_2class/results/"
echo "  CNN6:     ../cnn6/workspaces/full_ver_2class/results/"
echo ""

# 이메일 알림
python3 send_email.py "2-Class 비교 학습 완료" \
    "CNN6 vs ResNet22 2-class 공정 비교 완료 (149명 데이터)" 2>/dev/null || true
