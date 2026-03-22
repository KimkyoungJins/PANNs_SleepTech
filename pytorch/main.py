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

from utilities import create_folder, get_filename, create_logging
from models import Cnn14_16k
from pytorch_utils import move_data_to_device
from data_generator import SleepDataset, collate_fn
import config
from losses import get_loss_func


def train(args):
    """수면 단계 분류 모델 학습 함수."""

    data_dir = args.data_dir
    workspace = args.workspace
    pretrained_path = args.pretrained_path
    freeze_cnn = args.freeze_cnn
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    resume_path = args.resume_path  # 이어서 학습할 체크포인트
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = 16000
    window_size = 512
    hop_size = 160
    mel_bins = 64
    fmin = 50
    fmax = 8000
    classes_num = config.classes_num

    num_workers = 4

    # ===== 경로 설정 =====
    checkpoints_dir = os.path.join(workspace, 'checkpoints')
    create_folder(checkpoints_dir)

    logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU.')

    # ===== 1단계: 모델 생성 =====
    model = Cnn14_16k(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)

    # ===== Pretrained 가중치 로드 =====
    start_epoch = 0
    best_val_acc = 0.0

    if resume_path and os.path.exists(resume_path):
        # 이어서 학습 (기존 체크포인트에서 복원)
        logging.info('Resuming from checkpoint: {}'.format(resume_path))
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('val_acc', 0.0)
        logging.info('Resumed from epoch {}, val_acc {:.4f}'.format(start_epoch, best_val_acc))

    elif pretrained_path and os.path.exists(pretrained_path):
        logging.info('Loading pretrained weights from {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']

        model_dict = model.state_dict()
        pretrained_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_filtered[k] = v
            else:
                logging.info('  Skipped: {} (shape mismatch)'.format(k))

        model_dict.update(pretrained_filtered)
        model.load_state_dict(model_dict)
        logging.info('Loaded {} / {} pretrained layers'.format(
            len(pretrained_filtered), len(model_dict)))
    else:
        logging.info('No pretrained weights. Training from scratch.')

    # ===== CNN 동결 =====
    if freeze_cnn:
        logging.info('Freezing CNN layers')
        for name, param in model.named_parameters():
            if 'fc' in name or 'bn0' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info('Trainable: {} / {} parameters'.format(trainable, total))

    model.to(device)

    # ===== 2단계: 데이터 로더 =====
    train_dataset = SleepDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        audio_dir=data_dir,
        sample_rate=sample_rate)

    val_dataset = SleepDataset(
        csv_path=os.path.join(data_dir, 'val.csv'),
        audio_dir=data_dir,
        sample_rate=sample_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True)

    # ===== 3단계: 손실 함수 & 옵티마이저 =====
    loss_func = get_loss_func('clip_ce')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # resume 시 옵티마이저 상태 복원
    if resume_path and os.path.exists(resume_path):
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info('Optimizer state restored.')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # ===== 학습 기록 저장용 =====
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # ===== 4단계: 학습 루프 =====
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch_idx, batch_data_dict in enumerate(train_loader):
            waveform = move_data_to_device(batch_data_dict['waveform'], device)
            target = move_data_to_device(batch_data_dict['target'], device)

            output_dict = model(waveform, None)
            loss = loss_func(output_dict, {'target': target})

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

        val_acc, val_loss_avg = evaluate(model, val_loader, loss_func, device)

        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        # 기록 저장
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)

        logging.info(
            'Epoch {}/{} ({:.1f}s) - '
            'Train Loss: {:.4f}, Train Acc: {:.4f} - '
            'Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
                epoch + 1, num_epochs, epoch_time,
                train_loss_avg, train_acc,
                val_loss_avg, val_acc))

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc}

            checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info('Best model saved! (Val Acc: {:.4f})'.format(val_acc))

        # 주기적 체크포인트
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc}

            checkpoint_path = os.path.join(
                checkpoints_dir, 'epoch_{}.pth'.format(epoch + 1))
            torch.save(checkpoint, checkpoint_path)

    # 학습 기록 저장
    history_path = os.path.join(workspace, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logging.info('Training history saved to {}'.format(history_path))

    logging.info('Training complete. Best Val Acc: {:.4f}'.format(best_val_acc))


def evaluate(model, data_loader, loss_func, device):
    """모델 검증 — 정확도와 평균 손실 반환."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data_dict in data_loader:
            waveform = move_data_to_device(batch_data_dict['waveform'], device)
            target = move_data_to_device(batch_data_dict['target'], device)

            output_dict = model(waveform, None)
            loss = loss_func(output_dict, {'target': target})

            total_loss += loss.item()
            predictions = torch.argmax(output_dict['clipwise_output'], dim=1)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    return accuracy, avg_loss


def test(args):
    """test.csv로 최종 평가 + 시각 리포트 생성."""

    data_dir = args.data_dir
    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = 16000
    window_size = 512
    hop_size = 160
    mel_bins = 64
    fmin = 50
    fmax = 8000
    classes_num = config.classes_num
    batch_size = args.batch_size

    results_dir = os.path.join(workspace, 'results')
    create_folder(results_dir)

    logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='a')

    # ===== 모델 로드 =====
    model = Cnn14_16k(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)

    if not checkpoint_path:
        checkpoint_path = os.path.join(workspace, 'checkpoints', 'best_model.pth')

    logging.info('Loading model from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    logging.info('Model loaded (trained epoch: {}, val_acc: {:.4f})'.format(
        checkpoint.get('epoch', '?'), checkpoint.get('val_acc', 0)))

    # ===== 테스트 데이터 로드 =====
    test_dataset = SleepDataset(
        csv_path=os.path.join(data_dir, 'test.csv'),
        audio_dir=data_dir,
        sample_rate=sample_rate)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True)

    # ===== 추론 =====
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_data_dict in test_loader:
            waveform = move_data_to_device(batch_data_dict['waveform'], device)
            target = batch_data_dict['target']

            output_dict = model(waveform, None)
            probs = torch.softmax(output_dict['clipwise_output'], dim=1).cpu().numpy()
            predictions = np.argmax(probs, axis=1)

            all_predictions.extend(predictions)
            all_targets.extend(target)
            all_probs.extend(probs)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # ===== 결과 계산 =====
    accuracy = np.mean(all_predictions == all_targets)
    label_names = config.labels  # ['rem', 'nrem', 'wake']

    # Confusion Matrix
    cm = np.zeros((classes_num, classes_num), dtype=int)
    for t, p in zip(all_targets, all_predictions):
        cm[t][p] += 1

    # Per-class metrics
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

    # ===== 콘솔 출력 =====
    logging.info('\n' + '='*60)
    logging.info('TEST RESULTS')
    logging.info('='*60)
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

    # ===== 결과 JSON 저장 =====
    result_dict = {
        'accuracy': float(accuracy),
        'per_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_class.items()},
        'confusion_matrix': cm.tolist(),
        'label_names': label_names,
    }
    result_path = os.path.join(results_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logging.info('Results saved to {}'.format(result_path))

    # ===== 시각 리포트 생성 =====
    try:
        generate_report(workspace, results_dir, cm, per_class, label_names, accuracy)
        logging.info('Visual report saved to {}'.format(results_dir))
    except Exception as e:
        logging.warning('Could not generate visual report: {}'.format(e))
        logging.info('Install matplotlib: pip install matplotlib')


def generate_report(workspace, results_dir, cm, per_class, label_names, accuracy):
    """학습 곡선 + Confusion Matrix + 분류 리포트 시각화."""
    import matplotlib
    matplotlib.use('Agg')  # 서버에서도 동작하도록 GUI 없는 백엔드
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Sleep Stage Classification Report\nOverall Accuracy: {:.1f}%'.format(
        accuracy * 100), fontsize=16, fontweight='bold')

    # --- 1) 학습 곡선 (Loss) ---
    history_path = os.path.join(workspace, 'history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        ax = axes[0, 0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 2) 학습 곡선 (Accuracy) ---
        ax = axes[0, 1]
        ax.plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train Acc')
        ax.plot(epochs, [a * 100 for a in history['val_acc']], 'r-', label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training & Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training history', ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, 'No training history', ha='center', va='center')

    # --- 3) Confusion Matrix ---
    ax = axes[1, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # 숫자 표시
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color=color, fontsize=14)

    # --- 4) Per-class F1 Score ---
    ax = axes[1, 1]
    f1_scores = [per_class[n]['f1'] for n in label_names]
    bars = ax.bar(label_names, f1_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-class F1 Score')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                '{:.3f}'.format(score), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'report.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compare(args):
    """full_ver vs ratio_ver 두 모델 비교 리포트."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workspace_full = args.workspace_full
    workspace_ratio = args.workspace_ratio

    results = {}
    for name, ws in [('full_ver', workspace_full), ('ratio_ver', workspace_ratio)]:
        result_path = os.path.join(ws, 'results', 'test_results.json')
        if not os.path.exists(result_path):
            print(f'{name}: test_results.json not found at {result_path}')
            print(f'  Run test first: python3 main.py test --workspace={ws} ...')
            return
        with open(result_path) as f:
            results[name] = json.load(f)

    label_names = results['full_ver']['label_names']

    # 비교 출력
    print('\n' + '='*70)
    print('COMPARISON: full_ver vs ratio_ver')
    print('='*70)
    print('{:<12} {:>15} {:>15}'.format('', 'full_ver', 'ratio_ver'))
    print('-'*42)
    print('{:<12} {:>14.1f}% {:>14.1f}%'.format(
        'Accuracy',
        results['full_ver']['accuracy'] * 100,
        results['ratio_ver']['accuracy'] * 100))
    print()

    print('{:<12} {:>7} {:>7} {:>7} {:>7}'.format(
        '', 'F1(full)', 'F1(ratio)', 'Rec(full)', 'Rec(ratio)'))
    print('-'*50)
    for name in label_names:
        f1_full = results['full_ver']['per_class'][name]['f1']
        f1_ratio = results['ratio_ver']['per_class'][name]['f1']
        rec_full = results['full_ver']['per_class'][name]['recall']
        rec_ratio = results['ratio_ver']['per_class'][name]['recall']
        print('{:<12} {:>7.3f} {:>9.3f} {:>9.3f} {:>10.3f}'.format(
            name, f1_full, f1_ratio, rec_full, rec_ratio))

    # 비교 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('full_ver vs ratio_ver Comparison', fontsize=16, fontweight='bold')

    x = np.arange(len(label_names))
    width = 0.35

    # F1 Score 비교
    ax = axes[0]
    f1_full = [results['full_ver']['per_class'][n]['f1'] for n in label_names]
    f1_ratio = [results['ratio_ver']['per_class'][n]['f1'] for n in label_names]
    ax.bar(x - width/2, f1_full, width, label='full_ver', color='#4ecdc4')
    ax.bar(x + width/2, f1_ratio, width, label='ratio_ver', color='#ff6b6b')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Recall 비교
    ax = axes[1]
    rec_full = [results['full_ver']['per_class'][n]['recall'] for n in label_names]
    rec_ratio = [results['ratio_ver']['per_class'][n]['recall'] for n in label_names]
    ax.bar(x - width/2, rec_full, width, label='full_ver', color='#4ecdc4')
    ax.bar(x + width/2, rec_ratio, width, label='ratio_ver', color='#ff6b6b')
    ax.set_ylabel('Recall')
    ax.set_title('Recall by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Confusion Matrix 비교
    ax = axes[2]
    cm_full = np.array(results['full_ver']['confusion_matrix'])
    cm_ratio = np.array(results['ratio_ver']['confusion_matrix'])
    # 정규화된 정확도 차이
    cm_full_norm = cm_full / cm_full.sum(axis=1, keepdims=True)
    cm_ratio_norm = cm_ratio / cm_ratio.sum(axis=1, keepdims=True)
    diff = cm_ratio_norm - cm_full_norm

    im = ax.imshow(diff, interpolation='nearest', cmap='RdYlGn', vmin=-0.3, vmax=0.3)
    ax.set_title('Accuracy Diff (ratio - full)')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, '{:+.2f}'.format(diff[i][j]),
                    ha='center', va='center', fontsize=12)

    plt.tight_layout()
    compare_dir = os.path.join(os.path.dirname(workspace_full), 'comparison')
    create_folder(compare_dir)
    plt.savefig(os.path.join(compare_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print('\nComparison saved to {}'.format(compare_dir))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sleep Stage Classification with PANNs')
    subparsers = parser.add_subparsers(dest='mode')

    # ===== train =====
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--data_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--pretrained_path', type=str, default=None)
    parser_train.add_argument('--freeze_cnn', action='store_true', default=False)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--learning_rate', type=float, default=1e-4)
    parser_train.add_argument('--num_epochs', type=int, default=50)
    parser_train.add_argument('--resume_path', type=str, default=None,
        help='체크포인트 경로. 이어서 학습할 때 사용')
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # ===== test =====
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--data_dir', type=str, required=True)
    parser_test.add_argument('--workspace', type=str, required=True)
    parser_test.add_argument('--checkpoint_path', type=str, default=None,
        help='평가할 모델 경로. 없으면 best_model.pth 사용')
    parser_test.add_argument('--batch_size', type=int, default=16)
    parser_test.add_argument('--cuda', action='store_true', default=False)

    # ===== compare =====
    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument('--workspace_full', type=str, required=True)
    parser_compare.add_argument('--workspace_ratio', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'compare':
        compare(args)
    else:
        raise Exception('Use: python main.py [train|test|compare] ...')
