#!/usr/bin/env python3
"""Hierarchical 3-class 결과를 기존 workspace 형식의 report.png로 생성."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
VER_DIR = os.path.join(BASE_DIR, 'checkpoints', 'hierarchical', 'best')
RESULTS_DIR = os.path.join(VER_DIR, 'results')

# Load results
with open(os.path.join(RESULTS_DIR, 'test_balanced.json')) as f:
    results = json.load(f)

# Load EOG_CNN training history
eog_history_path = os.path.join(VER_DIR, 'training', 'stage2_history.json')
with open(eog_history_path) as f:
    eog_history = json.load(f)

# Load ResNet22 training history
resnet_history_path = os.path.join(VER_DIR, 'training', 'stage1_history.json')
with open(resnet_history_path) as f:
    resnet_history = json.load(f)

label_names = results['label_names']
cm = np.array(results['confusion_matrix'])
per_class = results['per_class']
accuracy = results['accuracy']
macro_f1 = results['macro_f1']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    f'Hierarchical 3-Class Sleep Stage Classification Report\n'
    f'Overall Accuracy: {accuracy*100:.1f}%  |  Macro F1: {macro_f1:.3f}  |  '
    f'Balanced Accuracy: {results["balanced_accuracy"]*100:.1f}%',
    fontsize=15, fontweight='bold'
)

# ── Panel 1: ResNet22 Training Curves ──
ax = axes[0, 0]
epochs_r = range(1, len(resnet_history['train_loss']) + 1)
ax.plot(epochs_r, resnet_history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
ax.plot(epochs_r, resnet_history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Stage 1: ResNet22 (Wake/Sleep) Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 2: EOG_CNN Training Curves ──
ax = axes[0, 1]
epochs_e = [h['epoch'] for h in eog_history]
train_loss_e = [h['train_loss'] for h in eog_history]
val_loss_e = [h['val_loss'] for h in eog_history]
val_macro_f1 = [h['val_macro_f1'] for h in eog_history]

ax2 = ax.twinx()
ax.plot(epochs_e, train_loss_e, 'b-', label='Train Loss', linewidth=1.5)
ax.plot(epochs_e, val_loss_e, 'r-', label='Val Loss', linewidth=1.5)
ax2.plot(epochs_e, val_macro_f1, 'g--', label='Val Macro F1', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax2.set_ylabel('Macro F1', color='g')
ax.set_title('Stage 2: EOG_CNN (REM/NREM) Loss & F1')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# ── Panel 3: Confusion Matrix ──
ax = axes[1, 0]
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('3-Class Confusion Matrix (Balanced Test)')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(label_names)))
ax.set_yticks(range(len(label_names)))
ax.set_xticklabels(label_names, fontsize=12)
ax.set_yticklabels(label_names, fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

for i in range(len(label_names)):
    for j in range(len(label_names)):
        color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                color=color, fontsize=14, fontweight='bold')

# ── Panel 4: Per-class Precision / Recall / F1 ──
ax = axes[1, 1]
x = np.arange(len(label_names))
width = 0.25

precisions = [per_class[n]['precision'] for n in label_names]
recalls = [per_class[n]['recall'] for n in label_names]
f1s = [per_class[n]['f1'] for n in label_names]

bars_p = ax.bar(x - width, precisions, width, label='Precision', color='#4ecdc4')
bars_r = ax.bar(x, recalls, width, label='Recall', color='#ff6b6b')
bars_f = ax.bar(x + width, f1s, width, label='F1', color='#45b7d1')

ax.set_ylabel('Score')
ax.set_title('Per-class Metrics')
ax.set_xticks(x)
ax.set_xticklabels(label_names, fontsize=12)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f'{h:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, 'report.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Report saved to {output_path}')
print(f'\nTest Results:')
print(f'  Accuracy: {accuracy*100:.1f}%')
print(f'  Macro F1: {macro_f1:.4f}')
print(f'  Stage 1 Acc: {results["stage1_accuracy"]*100:.1f}%')
print(f'  Stage 2 Acc: {results["stage2_accuracy"]*100:.1f}%')
for name in label_names:
    m = per_class[name]
    print(f'  {name:>5s}: P={m["precision"]:.4f} R={m["recall"]:.4f} F1={m["f1"]:.4f}')
