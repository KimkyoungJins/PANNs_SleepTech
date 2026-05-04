#!/bin/bash
# ============================================================
# PANNs ResNet22 (AudioSet pretrained) → 2048-d embedding 추출
#
# 한 번만 실행하면 됨. 결과는 bilstm/embeddings/ 에 저장.
# 149명 × ~1000 epoch ≈ 1.2GB 디스크 사용.
#
# 실행: cd project1/bilstm && bash scripts/1_extract_embeddings.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo ""
echo "============================================================"
echo "  PANNs ResNet22 embedding 추출 (149명)"
echo "  소스: ../resnet/ResNet22_mAP=0.430.pth (AudioSet 527 classes)"
echo "  출력: ./embeddings/{patient_id}_{emb,labels,meta}.{npy,json}"
echo "============================================================"
echo ""

python3 extract_embeddings.py \
    --data_dir=../data/data_for_ai/full_ver_3class \
    --pretrained_path=../resnet/ResNet22_mAP=0.430.pth \
    --output_dir=./embeddings \
    --batch_size=16 \
    --skip_existing \
    --cuda
