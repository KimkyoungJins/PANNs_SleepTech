#!/bin/bash
# Cnn6 — 2클래스만 (sleep / wake)

DATA_FULL="../data_for_ai/full_ver"
DATA_RATIO="../data_for_ai/ratio_ver"
PRETRAINED="./Cnn6_mAP=0.343.pth"
UTILS="./utils"

echo ""
echo "####################################################"
echo "  Cnn6: 2클래스 (sleep / wake)"
echo "####################################################"

cp $UTILS/config_2class.py $UTILS/config.py

python3 pytorch/main.py train \
    --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=5e-5 --num_epochs=50 \
    --loss_type=clip_ce --cuda

python3 pytorch/main.py train \
    --data_dir=$DATA_RATIO --workspace=./workspaces/ratio_ver_2class \
    --pretrained_path=$PRETRAINED --freeze_cnn \
    --batch_size=8 --learning_rate=5e-5 --num_epochs=50 \
    --loss_type=clip_ce --cuda

python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/full_ver_2class --batch_size=8 --cuda
python3 pytorch/main.py test --data_dir=$DATA_FULL --workspace=./workspaces/ratio_ver_2class --batch_size=8 --cuda
python3 pytorch/main.py compare --workspace_full=./workspaces/full_ver_2class --workspace_ratio=./workspaces/ratio_ver_2class

echo ""
echo "####################################################"
echo "  2클래스 완료!"
echo "####################################################"
echo ""
echo "결과 위치:"
echo "  full:  ./workspaces/full_ver_2class/results/report.png"
echo "  ratio: ./workspaces/ratio_ver_2class/results/report.png"

# 이메일 알림
python3 send_email.py "Cnn6 2클래스 학습 완료" "Cnn6 2클래스(sleep/wake) 학습이 완료되었습니다."
