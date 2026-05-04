#!/bin/bash
# ============================================================
# BiLSTM 평가 — 단독(Stage 2 only) + Hierarchical(Stage 1+2)
#
# 실행: cd project1/bilstm && bash scripts/3_eval.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

WORKSPACE="./workspaces/v1"
STAGE1_CKPT="../resnet/workspaces/full_ver_2class/checkpoints/best_model.pth"

echo ""
echo "############################################################"
echo "  [1/3] Validation 평가 (BiLSTM 단독)"
echo "############################################################"
python3 eval.py --workspace=$WORKSPACE --split=val --cuda

echo ""
echo "############################################################"
echo "  [2/3] Test 평가 (BiLSTM 단독)"
echo "############################################################"
python3 eval.py --workspace=$WORKSPACE --split=test --cuda

echo ""
echo "############################################################"
echo "  [3/4] Hierarchical 3-class 평가 (Stage 1 + Stage 2)"
echo "############################################################"
python3 hierarchical_eval.py \
    --bilstm_workspace=$WORKSPACE \
    --stage1_ckpt=$STAGE1_CKPT \
    --cuda

echo ""
echo "############################################################"
echo "  [4/4] 결과 시각화 (PNG 생성)"
echo "############################################################"
python3 visualize.py --workspace=$WORKSPACE

echo ""
echo "============================================================"
echo "  완료. 결과: $WORKSPACE/results/"
echo "    JSON:"
echo "      - eval_val.json"
echo "      - eval_test.json"
echo "      - hierarchical_3class.json"
echo "    PNG:"
echo "      - training_curves.png"
echo "      - val_report.png"
echo "      - test_report.png"
echo "      - hierarchical_report.png"
echo "============================================================"
