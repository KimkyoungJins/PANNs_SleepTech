#!/bin/bash
# ============================================================
# CNN6 vs ResNet22 — 2-Class Fine-tuning 비교
# 동결 해제 + lr=1e-5로 전체 모델 미세 조정
#
# 변경점 (10_compare_2class.sh 대비):
#   --freeze_cnn 제거 (CNN 동결 해제)
#   --learning_rate=1e-5 (1e-4 → 1e-5, pretrained 보호)
#   workspace: full_ver_2class_finetune (기존 결과 보존)
#
# 실행: cd project1/resnet && bash scripts/11_compare_2class_finetune.sh
# ============================================================

set -e

DATA="../data/data_for_ai/full_ver_2class"
UTILS="./utils"

echo ""
echo "============================================================"
echo "  CNN6 vs ResNet22 — 2-Class Fine-tuning 비교"
echo "  데이터: $DATA (149명, full_ver)"
echo "  조건: clip_ce, oversample, batch=8, lr=1e-5, patience=10"
echo "  변경: CNN 동결 해제 (전체 fine-tuning)"
echo "============================================================"

# ── config 설정 (2-class) ──
cp $UTILS/config_2class.py $UTILS/config.py

# ============================================================
#  1. ResNet22 Fine-tuning + 테스트
# ============================================================
echo ""
echo "####################################################"
echo "  [1/2] ResNet22 2-class Fine-tuning"
echo "####################################################"

python3 pytorch/main.py train \
    --data_dir=$DATA \
    --workspace=./workspaces/full_ver_2class_finetune \
    --pretrained_path=./ResNet22_mAP=0.430.pth \
    --batch_size=8 --learning_rate=1e-5 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

echo ""
echo ">>> ResNet22 Fine-tuning 테스트..."
python3 pytorch/main.py test \
    --data_dir=$DATA \
    --workspace=./workspaces/full_ver_2class_finetune \
    --batch_size=8 --cuda

# ============================================================
#  2. CNN6 Fine-tuning + 테스트
# ============================================================
echo ""
echo "####################################################"
echo "  [2/2] CNN6 2-class Fine-tuning"
echo "####################################################"

# CNN6 config도 2-class로 설정
cp ../cnn6/utils/config_2class.py ../cnn6/utils/config.py

python3 ../cnn6/pytorch/main.py train \
    --data_dir=$DATA \
    --workspace=../cnn6/workspaces/full_ver_2class_finetune \
    --pretrained_path=../cnn6/Cnn6_mAP=0.343.pth \
    --batch_size=8 --learning_rate=1e-5 --num_epochs=50 \
    --loss_type=clip_ce --oversample --patience=10 --cuda

echo ""
echo ">>> CNN6 Fine-tuning 테스트..."
python3 ../cnn6/pytorch/main.py test \
    --data_dir=$DATA \
    --workspace=../cnn6/workspaces/full_ver_2class_finetune \
    --batch_size=8 --cuda

# ============================================================
#  3. 결과 비교 (동결 vs Fine-tuning 4가지 모두)
# ============================================================
echo ""
echo "####################################################"
echo "  결과 비교 (동결 vs Fine-tuning)"
echo "####################################################"

python3 -c "
import json, os

models = {
    'ResNet22 (동결)':      './workspaces/full_ver_2class/results/test_results.json',
    'ResNet22 (finetune)':  './workspaces/full_ver_2class_finetune/results/test_results.json',
    'CNN6 (동결)':          '../cnn6/workspaces/full_ver_2class/results/test_results.json',
    'CNN6 (finetune)':      '../cnn6/workspaces/full_ver_2class_finetune/results/test_results.json',
}

print()
print('=' * 80)
print('  동결 vs Fine-tuning 비교 결과 (149명 데이터, balanced test)')
print('=' * 80)
print(f'{\"\":>22s} {\"Accuracy\":>10s} {\"Wake P\":>8s} {\"Wake R\":>8s} {\"Wake F1\":>8s} {\"Sleep P\":>8s} {\"Sleep R\":>8s} {\"Sleep F1\":>8s}')
print('-' * 80)

for name, path in models.items():
    if not os.path.exists(path):
        print(f'{name:>22s}  (결과 없음)')
        continue
    with open(path) as f:
        r = json.load(f)
    w = r['per_class']['wake']
    s = r['per_class']['sleep']
    print(f'{name:>22s} {r[\"accuracy\"]*100:>9.2f}% {w[\"precision\"]:>8.4f} {w[\"recall\"]:>8.4f} {w[\"f1\"]:>8.4f} {s[\"precision\"]:>8.4f} {s[\"recall\"]:>8.4f} {s[\"f1\"]:>8.4f}')

print()
"

echo ""
echo "============================================================"
echo "  완료!"
echo "============================================================"
echo ""
echo "결과:"
echo "  ResNet22 동결:     ./workspaces/full_ver_2class/results/"
echo "  ResNet22 finetune: ./workspaces/full_ver_2class_finetune/results/"
echo "  CNN6 동결:         ../cnn6/workspaces/full_ver_2class/results/"
echo "  CNN6 finetune:     ../cnn6/workspaces/full_ver_2class_finetune/results/"
echo ""

# 이메일 알림
python3 send_email.py "Fine-tuning 비교 완료" \
    "CNN6 vs ResNet22 동결 vs Fine-tuning 비교 완료 (149명 데이터)" 2>/dev/null || true
