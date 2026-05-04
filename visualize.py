#!/usr/bin/env python3
"""
BiLSTM 결과 시각화.

생성 파일 (workspaces/{name}/results/):
  - training_curves.png    : loss + macro F1 / balanced acc 학습 곡선
  - test_report.png        : test 평가 (REM/NREM 단독, per-window + per-epoch)
  - hierarchical_report.png: 3-class hierarchical (Wake/REM/NREM)

사용:
    python3 visualize.py --workspace=./workspaces/v1
    python3 visualize.py --workspace=./workspaces/v1 --skip_missing
"""

import os
import sys
import json
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── 색상 (기존 report.html 과 일관) ──
COLOR_REM = '#ff6b6b'      # 코랄
COLOR_NREM = '#4ecdc4'     # 청록
COLOR_WAKE = '#f9ca24'     # 노랑
COLOR_TRAIN = '#2980b9'    # 진파랑
COLOR_VAL = '#e74c3c'      # 빨강
COLOR_BASELINE = '#7f8c8d' # 회색 (v1 baseline)

# v1 EOG_CNN baseline (model_results_v1.md 기준)
V1_BASELINE = {
    '3class_balanced_acc': 0.737,
    '3class_macro_f1': 0.7392,
    'wake':  {'precision': 0.803, 'recall': 0.754, 'f1': 0.778},
    'rem':   {'precision': 0.870, 'recall': 0.639, 'f1': 0.736},
    'nrem':  {'precision': 0.617, 'recall': 0.819, 'f1': 0.704},
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def annotate_cm(ax, cm, labels, normalize=False):
    """Confusion matrix heatmap에 숫자 라벨 + 색상 차별."""
    cm = np.asarray(cm)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum
    else:
        cm_norm = cm
    im = ax.imshow(cm_norm if normalize else cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    threshold = (cm.max() if not normalize else 0.5) / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            color = 'white' if (cm_norm[i, j] if normalize else v) > threshold else 'black'
            txt = f'{v}' if not normalize else f'{cm_norm[i,j]:.2f}\n({v})'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=11)


# ──────────────────────────────────────────────────────
# 1. 학습 곡선
# ──────────────────────────────────────────────────────
def plot_training_curves(workspace, out_path):
    history = load_json(os.path.join(workspace, 'history.json'))
    if history is None:
        print(f'  [SKIP] history.json 없음 → {out_path}')
        return False

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_f1 = [h['val_macro_f1'] for h in history]
    val_bal = [h['val_balanced_acc'] for h in history]
    lrs = [h.get('lr', np.nan) for h in history]

    best_idx = int(np.argmax(val_f1))
    best_epoch = epochs[best_idx]
    best_f1 = val_f1[best_idx]
    best_bal = val_bal[best_idx]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    fig.suptitle(
        f'BiLSTM REM/NREM Training Curves\n'
        f'Best Epoch {best_epoch}: val_macro_F1={best_f1:.4f}, val_BalAcc={best_bal:.4f}',
        fontsize=15, fontweight='bold'
    )

    # (0,0) Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_loss, color=COLOR_TRAIN, label='Train', linewidth=2)
    ax.plot(epochs, val_loss, color=COLOR_VAL, label='Val', linewidth=2)
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Focal Loss')
    ax.set_title('Train vs Val Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) Val Macro F1 + Val Balanced Acc
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, val_f1, color=COLOR_REM, marker='o', markersize=3,
            label='Val Macro F1', linewidth=2)
    ax.plot(epochs, val_bal, color=COLOR_NREM, marker='s', markersize=3,
            label='Val Balanced Acc', linewidth=2)
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.6)
    ax.scatter([best_epoch], [best_f1], color='green', s=80, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # (1,0) Learning rate
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, lrs, color='purple', marker='.', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # (1,1) Per-class recall over epochs (REM_R, NREM_R from confusion matrix)
    ax = fig.add_subplot(gs[1, 1])
    rem_r, nrem_r = [], []
    for h in history:
        cm = np.array(h.get('val_confusion_matrix', [[0, 0], [0, 0]]))
        rem_r.append(cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0)
        nrem_r.append(cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0)
    ax.plot(epochs, rem_r, color=COLOR_REM, label='REM Recall', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, nrem_r, color=COLOR_NREM, label='NREM Recall', linewidth=2, marker='s', markersize=3)
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall (Validation)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out_path}')
    return True


# ──────────────────────────────────────────────────────
# 2. Test 결과 (Stage 2 단독, REM/NREM)
# ──────────────────────────────────────────────────────
def plot_test_report(workspace, out_path, split='test'):
    eval_path = os.path.join(workspace, 'results', f'eval_{split}.json')
    data = load_json(eval_path)
    if data is None:
        print(f'  [SKIP] {eval_path} 없음 → {out_path}')
        return False

    pw = data['per_window']
    pe = data['per_epoch']
    labels = ['REM', 'NREM']

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.3)
    fig.suptitle(
        f'BiLSTM REM/NREM — {split.upper()} Evaluation (Stage 2 only)\n'
        f'Per-Epoch (sliding aggregated): BalAcc={pe["balanced_acc"]:.4f}, '
        f'MacroF1={pe["macro_f1"]:.4f}',
        fontsize=15, fontweight='bold'
    )

    # (0,0) per-window CM
    ax = fig.add_subplot(gs[0, 0])
    annotate_cm(ax, pw['confusion_matrix'], labels, normalize=False)
    ax.set_title(f'Per-Window CM (center {data.get("per_window_center","20")})\n'
                 f'BalAcc={pw["balanced_acc"]:.4f}, F1={pw["macro_f1"]:.4f}')

    # (0,1) per-epoch CM (정규화)
    ax = fig.add_subplot(gs[0, 1])
    annotate_cm(ax, pe['confusion_matrix'], labels, normalize=True)
    ax.set_title(f'Per-Epoch CM (normalized by row)\n'
                 f'BalAcc={pe["balanced_acc"]:.4f}, F1={pe["macro_f1"]:.4f}')

    # (1,0) per-class metrics bar (per-epoch)
    ax = fig.add_subplot(gs[1, 0])
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    rem_vals = [pe['rem'][m] for m in metrics]
    nrem_vals = [pe['nrem'][m] for m in metrics]
    bars1 = ax.bar(x - width/2, rem_vals, width, label='REM', color=COLOR_REM)
    bars2 = ax.bar(x + width/2, nrem_vals, width, label='NREM', color=COLOR_NREM)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics (Per-Epoch)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')

    # (1,1) per-window vs per-epoch 비교
    ax = fig.add_subplot(gs[1, 1])
    eval_modes = ['Per-Window', 'Per-Epoch']
    bal_accs = [pw['balanced_acc'], pe['balanced_acc']]
    f1s = [pw['macro_f1'], pe['macro_f1']]
    x = np.arange(len(eval_modes))
    width = 0.35
    bars1 = ax.bar(x - width/2, bal_accs, width, label='Balanced Acc', color='#3498db')
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1', color='#9b59b6')
    ax.set_xticks(x)
    ax.set_xticklabels(eval_modes)
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Mode Comparison')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out_path}')
    return True


# ──────────────────────────────────────────────────────
# 3. Hierarchical 3-class 결과
# ──────────────────────────────────────────────────────
def plot_hierarchical_report(workspace, out_path):
    h_path = os.path.join(workspace, 'results', 'hierarchical_3class.json')
    data = load_json(h_path)
    if data is None:
        print(f'  [SKIP] {h_path} 없음 → {out_path}')
        return False

    labels = ['Wake', 'REM', 'NREM']
    colors = [COLOR_WAKE, COLOR_REM, COLOR_NREM]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.28)
    fig.suptitle(
        f'Hierarchical 3-Class Result (Stage1 ResNet22 + Stage2 BiLSTM)\n'
        f'Acc={data["accuracy"]*100:.2f}%, BalAcc={data["balanced_accuracy"]*100:.2f}%, '
        f'MacroF1={data["macro_f1"]:.4f}  '
        f'(v1 EOG_CNN: BalAcc=73.7%, F1=0.7392)',
        fontsize=14, fontweight='bold'
    )

    # (0,0) 3x3 CM (count)
    ax = fig.add_subplot(gs[0, 0])
    annotate_cm(ax, data['confusion_matrix'], labels, normalize=False)
    ax.set_title('Confusion Matrix (count)')

    # (0,1) 3x3 CM (normalized)
    ax = fig.add_subplot(gs[0, 1])
    annotate_cm(ax, data['confusion_matrix'], labels, normalize=True)
    ax.set_title('Confusion Matrix (normalized by row)')

    # (1,0) Per-class F1 vs v1 baseline
    ax = fig.add_subplot(gs[1, 0])
    metrics_to_plot = ['precision', 'recall', 'f1']
    x = np.arange(len(labels))
    width = 0.13
    metric_colors = {'precision': '#3498db', 'recall': '#e67e22', 'f1': '#9b59b6'}

    # v2 (ours)
    for mi, m in enumerate(metrics_to_plot):
        v2_vals = [data['per_class'][lb.lower()][m] for lb in labels]
        offset = (mi - 1) * width * 2
        bars = ax.bar(x + offset - width, v2_vals, width,
                      label=f'v2 {m.capitalize()}',
                      color=metric_colors[m], edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, v2_vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    f'{v:.2f}', ha='center', fontsize=8)

    # v1 baseline (faded bars)
    for mi, m in enumerate(metrics_to_plot):
        v1_vals = [V1_BASELINE[lb.lower()][m] for lb in labels]
        offset = (mi - 1) * width * 2
        ax.bar(x + offset, v1_vals, width,
               label=f'v1 {m.capitalize()}',
               color=metric_colors[m], alpha=0.35, hatch='//',
               edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics (v2 BiLSTM solid vs v1 EOG_CNN hatched)')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # (1,1) Per-patient accuracy 분포
    ax = fig.add_subplot(gs[1, 1])
    pp = data.get('per_patient', {})
    if pp:
        accs = [v['accuracy'] for v in pp.values()]
        ax.hist(accs, bins=15, color='#34495e', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(accs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(accs):.3f}')
        ax.axvline(np.median(accs), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(accs):.3f}')
        ax.set_xlabel('Per-Patient Accuracy')
        ax.set_ylabel('Number of Patients')
        ax.set_title(f'Per-Patient Accuracy Distribution (N={len(accs)})')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No per-patient data', ha='center', va='center')
        ax.axis('off')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out_path}')
    return True


# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='./workspaces/v1')
    parser.add_argument('--skip_missing', action='store_true',
                        help='파일 없어도 에러 없이 스킵')
    args = parser.parse_args()

    results_dir = os.path.join(args.workspace, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f'Workspace: {args.workspace}')
    print(f'Output dir: {results_dir}')
    print()

    ok = []
    print('[1/4] Training curves')
    ok.append(plot_training_curves(
        args.workspace, os.path.join(results_dir, 'training_curves.png')))

    print('[2/4] Val report (REM/NREM, Stage 2 only)')
    ok.append(plot_test_report(
        args.workspace, os.path.join(results_dir, 'val_report.png'), split='val'))

    print('[3/4] Test report (REM/NREM, Stage 2 only)')
    ok.append(plot_test_report(
        args.workspace, os.path.join(results_dir, 'test_report.png'), split='test'))

    print('[4/4] Hierarchical 3-class report')
    ok.append(plot_hierarchical_report(
        args.workspace, os.path.join(results_dir, 'hierarchical_report.png')))

    n_ok = sum(ok)
    print()
    print(f'생성: {n_ok}/4')
    if n_ok < 4 and not args.skip_missing:
        print('일부 결과 파일이 없습니다. 학습/평가 후 다시 실행하세요.')


if __name__ == '__main__':
    main()
