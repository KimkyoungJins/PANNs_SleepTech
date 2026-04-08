"""
CNN(ResNet22) + Bi-LSTM 수면 단계 분류 — 학습/테스트
기존 CNN 가중치를 로드하고, LSTM + FC만 학습합니다.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from utilities import create_folder, create_logging
from models_lstm import SleepStageLSTM
from pytorch_utils import move_data_to_device
from data_generator_lstm import SleepSequenceDataset, collate_fn_lstm
import config


def train(args):
    """LSTM 학습"""
    data_dir = args.data_dir
    workspace = args.workspace
    cnn_checkpoint = args.cnn_checkpoint
    seq_len = args.seq_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = args.classes_num
    config.classes_num = classes_num
    if classes_num == 2:
        config.labels = ['wake', 'sleep']
    else:
        config.labels = ['wake', 'rem', 'nrem']

    checkpoints_dir = os.path.join(workspace, 'checkpoints')
    create_folder(checkpoints_dir)
    logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='w')
    logging.info(args)
    logging.info('Using {}'.format(device))

    # ===== 모델 생성 =====
    model = SleepStageLSTM(
        classes_num=classes_num,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
    )

    # CNN 가중치 로드 + Freeze
    logging.info('Loading CNN weights from: {}'.format(cnn_checkpoint))
    model.load_cnn_weights(cnn_checkpoint, device=device)
    model.to(device)

    # ===== 데이터 로더 =====
    train_dataset = SleepSequenceDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        audio_dir=data_dir,
        seq_len=seq_len)

    val_dataset = SleepSequenceDataset(
        csv_path=os.path.join(data_dir, 'val.csv'),
        audio_dir=data_dir,
        seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_lstm,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_lstm,
        num_workers=4,
        pin_memory=True)

    # ===== 손실 함수 & 옵티마이저 =====
    loss_func = nn.CrossEntropyLoss()

    # LSTM + FC 파라미터만 학습 (CNN은 Freeze)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    # ===== 학습 기록 =====
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    # ===== 학습 루프 =====
    for epoch in range(num_epochs):
        model.train()
        model.cnn.eval()  # CNN은 항상 eval (BN, Dropout 고정)

        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch_idx, batch_data_dict in enumerate(train_loader):
            # [B, seq_len, 480000]
            waveforms = move_data_to_device(batch_data_dict['waveform'], device)
            target = move_data_to_device(batch_data_dict['target'], device)

            output_dict = model(waveforms)
            loss = loss_func(output_dict['clipwise_output'], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(output_dict['clipwise_output'], dim=1)
            train_correct += (predictions == target).sum().item()
            train_total += target.size(0)

            if batch_idx % 10 == 0:
                print('  Epoch {}, Batch {}/{}, Loss: {:.4f}'.format(
                    epoch + 1, batch_idx, len(train_loader), loss.item()))

        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # 검증
        val_acc, val_loss_avg = evaluate(model, val_loader, loss_func, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)

        logging.info(
            'Epoch {}/{} ({:.1f}s) - '
            'Train Loss: {:.4f}, Train Acc: {:.4f} - '
            'Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
                epoch + 1, num_epochs, epoch_time,
                train_loss_avg, train_acc, val_loss_avg, val_acc))

        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'seq_len': seq_len,
                'lstm_hidden': args.lstm_hidden,
                'lstm_layers': args.lstm_layers,
            }
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'best_model.pth'))
            logging.info('Best model saved! (Val Acc: {:.4f})'.format(val_acc))

        # 매 10 에폭 저장
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'seq_len': seq_len,
            }
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'epoch_{}.pth'.format(epoch + 1)))

    # 학습 기록 저장
    with open(os.path.join(workspace, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logging.info('Training complete. Best Val Acc: {:.4f}'.format(best_val_acc))


def evaluate(model, data_loader, loss_func, device):
    """모델 검증"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data_dict in data_loader:
            waveforms = move_data_to_device(batch_data_dict['waveform'], device)
            target = move_data_to_device(batch_data_dict['target'], device)

            output_dict = model(waveforms)
            loss = loss_func(output_dict['clipwise_output'], target)

            total_loss += loss.item()
            predictions = torch.argmax(output_dict['clipwise_output'], dim=1)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return accuracy, avg_loss


def test(args):
    """LSTM 테스트"""
    data_dir = args.data_dir
    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    seq_len = args.seq_len
    batch_size = args.batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = args.classes_num
    config.classes_num = classes_num
    if classes_num == 2:
        config.labels = ['wake', 'sleep']
    else:
        config.labels = ['wake', 'rem', 'nrem']

    results_dir = os.path.join(workspace, 'results')
    create_folder(results_dir)
    logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='a')

    # 체크포인트에서 하이퍼파라미터 읽기
    if not checkpoint_path:
        checkpoint_path = os.path.join(workspace, 'checkpoints', 'best_model.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lstm_hidden = checkpoint.get('lstm_hidden', 256)
    lstm_layers = checkpoint.get('lstm_layers', 2)
    saved_seq_len = checkpoint.get('seq_len', seq_len)

    logging.info('Loading LSTM model from {}'.format(checkpoint_path))
    logging.info('  seq_len={}, lstm_hidden={}, lstm_layers={}'.format(
        saved_seq_len, lstm_hidden, lstm_layers))

    # 모델 생성 + 가중치 로드
    model = SleepStageLSTM(
        classes_num=classes_num,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 데이터 로드
    test_dataset = SleepSequenceDataset(
        csv_path=os.path.join(data_dir, 'test.csv'),
        audio_dir=data_dir,
        seq_len=saved_seq_len)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_lstm,
        num_workers=4,
        pin_memory=True)

    # 추론
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data_dict in test_loader:
            waveforms = move_data_to_device(batch_data_dict['waveform'], device)
            target = batch_data_dict['target']

            output_dict = model(waveforms)
            probs = torch.softmax(output_dict['clipwise_output'], dim=1).cpu().numpy()
            predictions = np.argmax(probs, axis=1)

            all_predictions.extend(predictions)
            all_targets.extend(target)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 결과 계산
    accuracy = np.mean(all_predictions == all_targets)
    label_names = config.labels

    cm = np.zeros((classes_num, classes_num), dtype=int)
    for t, p in zip(all_targets, all_predictions):
        cm[t][p] += 1

    per_class = {}
    for i, name in enumerate(label_names):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(classes_num)) - tp
        fn = sum(cm[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(cm[i])
        per_class[name] = {
            'precision': precision, 'recall': recall,
            'f1': f1, 'support': support
        }

    # 출력
    logging.info('\n' + '=' * 60)
    logging.info('LSTM TEST RESULTS (seq_len={})'.format(saved_seq_len))
    logging.info('=' * 60)
    logging.info('Overall Accuracy: {:.1f}%'.format(accuracy * 100))
    logging.info('')
    logging.info('{:<10} {:>10} {:>10} {:>10} {:>10}'.format(
        '', 'precision', 'recall', 'f1-score', 'support'))
    for name in label_names:
        m = per_class[name]
        logging.info('{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10d}'.format(
            name, m['precision'], m['recall'], m['f1'], m['support']))

    logging.info('')
    logging.info('Confusion Matrix:')
    header = '{:<10}'.format('') + ''.join('{:>10}'.format('pred_' + n) for n in label_names)
    logging.info(header)
    for i, name in enumerate(label_names):
        row = '{:<10}'.format(name) + ''.join('{:>10d}'.format(cm[i][j]) for j in range(classes_num))
        logging.info(row)

    # 결과 저장
    result_dict = {
        'accuracy': float(accuracy),
        'seq_len': saved_seq_len,
        'per_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_class.items()},
        'confusion_matrix': cm.tolist(),
        'label_names': label_names,
    }
    result_path = os.path.join(results_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logging.info('Results saved to {}'.format(result_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN+LSTM Sleep Stage Classification')
    subparsers = parser.add_subparsers(dest='mode')

    # ===== train =====
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--data_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--cnn_checkpoint', type=str, required=True,
        help='기존 CNN best_model.pth 경로')
    parser_train.add_argument('--seq_len', type=int, default=10,
        help='연속 에포크 수 (기본 10 = 5분)')
    parser_train.add_argument('--lstm_hidden', type=int, default=256)
    parser_train.add_argument('--lstm_layers', type=int, default=2)
    parser_train.add_argument('--lstm_dropout', type=float, default=0.3)
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--learning_rate', type=float, default=1e-4)
    parser_train.add_argument('--num_epochs', type=int, default=50)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--classes_num', type=int, default=2,
        help='클래스 수 (2 또는 3)')

    # ===== test =====
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--data_dir', type=str, required=True)
    parser_test.add_argument('--workspace', type=str, required=True)
    parser_test.add_argument('--checkpoint_path', type=str, default=None)
    parser_test.add_argument('--seq_len', type=int, default=10)
    parser_test.add_argument('--batch_size', type=int, default=4)
    parser_test.add_argument('--cuda', action='store_true', default=False)
    parser_test.add_argument('--classes_num', type=int, default=2,
        help='클래스 수 (2 또는 3)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise Exception('Use: python main_lstm.py [train|test] ...')
