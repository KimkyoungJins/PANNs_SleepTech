#!/bin/bash
# Sweep 완료 대기 → EOG 변종 3개 순차 학습 → 각 변종으로 hier eval → best 결정
set -e

cd /home/sleeptech/sleeptech/project3/resnet
source /home/sleeptech/miniconda3/etc/profile.d/conda.sh
conda activate env_p1

log() { echo "[eog_chain $(date '+%H:%M:%S')] $*"; }

# 1. Sweep 완료 대기 — "Done." 라인 (스크립트 마지막)이 나타날 때까지
log "Waiting for threshold sweep to complete..."
while ! grep -qE "^Done\.$" /tmp/v3_threshold_sweep.log 2>/dev/null; do
    sleep 30
done
log "Threshold sweep done."

# 2. EOG 변종 3개 순차 학습 (모두 GPU 0)
log "Stage 2: EOG_CNN variants sequential training..."

log "  [1/3] eog_a085_g30 (alpha=0.85, gamma=3.0)"
CUDA_VISIBLE_DEVICES=0 python3 -u train_eog_cnn.py --version eog_a085_g30 --focal_alpha 0.85 --focal_gamma 3.0 > /tmp/eog_a085_g30.log 2>&1
log "  [1/3] done."

log "  [2/3] eog_a095_g30 (alpha=0.95, gamma=3.0)"
CUDA_VISIBLE_DEVICES=0 python3 -u train_eog_cnn.py --version eog_a095_g30 --focal_alpha 0.95 --focal_gamma 3.0 > /tmp/eog_a095_g30.log 2>&1
log "  [2/3] done."

log "  [3/3] eog_a075_g20 (alpha=0.75, gamma=2.0)"
CUDA_VISIBLE_DEVICES=0 python3 -u train_eog_cnn.py --version eog_a075_g20 --focal_alpha 0.75 --focal_gamma 2.0 > /tmp/eog_a075_g20.log 2>&1
log "  [3/3] done."

# 3. 각 변종의 best val_macro_f1 추출
log "Stage 3: Compare EOG variants..."
python3 << 'EOF'
import json
import os

variants = ['eog_a085_g30', 'eog_a095_g30', 'eog_a075_g20']
base = '/home/sleeptech/sleeptech/project3/resnet/checkpoints/hierarchical'

print(f"{'Variant':<20s} {'epochs':>6s} {'best_epoch':>10s} {'best_macro_f1':>12s} {'best_bal_acc':>12s}")
results = []
for v in variants:
    history_path = os.path.join(base, v, 'training', 'stage2_history.json')
    if not os.path.exists(history_path):
        print(f"{v:<20s}  no history")
        continue
    with open(history_path) as f:
        h = json.load(f)
    best_f1 = max(h, key=lambda e: e.get('val_macro_f1', -1))
    print(f"{v:<20s} {len(h):>6d} {best_f1['epoch']:>10d} "
          f"{best_f1['val_macro_f1']:>12.4f} {best_f1.get('val_bal_acc', 0):>12.4f}")
    results.append((v, best_f1['val_macro_f1']))

# Pick best
best_variant = max(results, key=lambda r: r[1])
print(f"\nBest variant: {best_variant[0]} (macro_f1={best_variant[1]:.4f})")
with open('/tmp/best_eog_variant.txt', 'w') as f:
    f.write(best_variant[0])
EOF

BEST_VARIANT=$(cat /tmp/best_eog_variant.txt)
log "Best EOG variant: $BEST_VARIANT"

# 4. Best EOG로 새 v3_user_proposed_v2 만들고 hier eval (full + balanced)
log "Stage 4: Create v3_user_proposed_v2 with best Stage 1 + best new EOG..."
NEW_VER=v4_user_proposed_best_eog
mkdir -p checkpoints/hierarchical/$NEW_VER
cp ../resnet/workspaces/full_ver_2class_finetune/checkpoints/best_model.pth \
   checkpoints/hierarchical/$NEW_VER/stage1_resnet22.pth
cp checkpoints/hierarchical/$BEST_VARIANT/stage2_eog_cnn.pth \
   checkpoints/hierarchical/$NEW_VER/stage2_eog_cnn.pth
echo "Stage 1: full_ver_2class_finetune (val_acc 0.8421)" > checkpoints/hierarchical/$NEW_VER/stage1_source.txt
echo "Stage 2: $BEST_VARIANT (best EOG variant)" > checkpoints/hierarchical/$NEW_VER/stage2_source.txt

log "Stage 5: hierarchical_threshold_sweep on $NEW_VER..."
CUDA_VISIBLE_DEVICES=0 python3 -u hierarchical_threshold_sweep.py --version $NEW_VER --test_type both 2>&1 | tee /tmp/v4_sweep.log
log "Done. New best: checkpoints/hierarchical/$NEW_VER/"
