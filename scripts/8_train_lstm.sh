#!/bin/bash
# CNN(ResNet22) + Bi-LSTM — full_ver 2클래스 + 3클래스 학습/테스트

DATA_FULL="../data/data_for_ai/full_ver"
UTILS="./utils"

# ======================================================
#  PART 1: 2클래스 LSTM (sleep / wake)
# ======================================================
echo ""
echo "####################################################"
echo "  PART 1: LSTM 2클래스 (sleep / wake)"
echo "####################################################"

cp $UTILS/config_2class.py $UTILS/config.py

python3 pytorch/main_lstm.py train \
    --data_dir=$DATA_FULL \
    --workspace=./workspaces/lstm_full_ver_2class \
    --cnn_checkpoint=./workspaces/full_ver_2class/checkpoints/best_model.pth \
    --seq_len=10 \
    --lstm_hidden=256 --lstm_layers=2 --lstm_dropout=0.3 \
    --batch_size=4 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=weighted_ce --oversample \
    --classes_num=2 --cuda

python3 pytorch/main_lstm.py test \
    --data_dir=$DATA_FULL \
    --workspace=./workspaces/lstm_full_ver_2class \
    --batch_size=4 --classes_num=2 --cuda

# ======================================================
#  PART 2: 3클래스 LSTM (rem / nrem / wake)
# ======================================================
echo ""
echo "####################################################"
echo "  PART 2: LSTM 3클래스 (rem / nrem / wake)"
echo "####################################################"

cp $UTILS/config_3class.py $UTILS/config.py

python3 pytorch/main_lstm.py train \
    --data_dir=$DATA_FULL \
    --workspace=./workspaces/lstm_full_ver_3class \
    --cnn_checkpoint=./workspaces/full_ver_3class/checkpoints/best_model.pth \
    --seq_len=10 \
    --lstm_hidden=256 --lstm_layers=2 --lstm_dropout=0.3 \
    --batch_size=4 --learning_rate=1e-4 --num_epochs=50 \
    --loss_type=weighted_ce --oversample \
    --classes_num=3 --cuda

python3 pytorch/main_lstm.py test \
    --data_dir=$DATA_FULL \
    --workspace=./workspaces/lstm_full_ver_3class \
    --batch_size=4 --classes_num=3 --cuda

# ======================================================
#  복원 & 완료
# ======================================================
cp $UTILS/config_2class.py $UTILS/config.py

echo ""
echo "####################################################"
echo "  LSTM 전체 완료!"
echo "####################################################"
echo ""
echo "결과 위치:"
echo "  2클래스 LSTM: ./workspaces/lstm_full_ver_2class/results/"
echo "  3클래스 LSTM: ./workspaces/lstm_full_ver_3class/results/"

# 이메일 알림
python3 send_email.py "LSTM 학습 완료" "CNN+LSTM 2클래스+3클래스 full_ver 학습이 완료되었습니다."
