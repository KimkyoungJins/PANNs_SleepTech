# 2026-05-03 — Phase 2 & 3 완료 (모든 코드 작성)

## ✅ 오늘 추가로 완료한 것

`extract_embeddings.py`(Phase 1) 작성에 이어, **나머지 모든 학습/평가 코드 작성**.
GPU 환경 준비되면 바로 실행 가능한 상태.

### 작성된 파일 (총 7개)

| 단계 | 파일 | 줄 수 | 용도 |
|---|---|---|---|
| Phase 1 | `extract_embeddings.py` | 215 | wav → ResNet22 → 2048-d .npy 캐시 |
| Phase 2 | `models/bilstm_classifier.py` | 100 | BiLSTM 모델 (1.32M params) |
| Phase 2 | `utils/losses.py` | 100 | MaskedFocalLoss (Wake mask 처리) |
| Phase 2 | `datasets/seq_dataset.py` | 240 | 환자별 40-epoch sequence Dataset |
| Phase 3 | `train.py` | 240 | 학습 루프 (Adam, ReduceLROnPlateau, early stopping) |
| Phase 3 | `eval.py` | 195 | Stage 2 단독 평가 (per-window + per-epoch) |
| Phase 3 | `hierarchical_eval.py` | 240 | Stage 1+2 통합 → 3-class 평가 |

### 실행 스크립트 (4개)

- `scripts/1_extract_embeddings.sh` — Phase 1 실행
- `scripts/2_train.sh` — Phase 3 학습
- `scripts/3_eval.sh` — Phase 3 평가 (val + test + hierarchical)
- `scripts/run_all.sh` — 전체 파이프라인 1회 실행

---

## 🏗️ 최종 구조

```
project1/bilstm/
├── extract_embeddings.py       ✓ wav → 2048-d
├── models/
│   ├── __init__.py
│   └── bilstm_classifier.py    ✓ BiLSTMClassifier
├── datasets/
│   ├── __init__.py
│   └── seq_dataset.py          ✓ SoundSequenceDataset, remap_labels
├── utils/
│   ├── __init__.py
│   └── losses.py               ✓ MaskedFocalLoss
├── train.py                    ✓ 학습 루프
├── eval.py                     ✓ Stage 2 단독 평가
├── hierarchical_eval.py        ✓ Stage 1+2 → 3-class
├── scripts/
│   ├── 1_extract_embeddings.sh
│   ├── 2_train.sh
│   ├── 3_eval.sh
│   └── run_all.sh
├── embeddings/                 (Phase 1 실행 후 .npy 채워짐)
├── checkpoints/                (이번엔 workspaces/v1 로 이동, 비어있음)
├── workspaces/
│   └── v1/                     (학습 후 채워짐)
│       ├── config.json
│       ├── history.json
│       ├── checkpoints/{best,last}_model.pth
│       ├── logs/train_*.log
│       └── results/{eval_val.json, eval_test.json, hierarchical_3class.json}
└── record/
    ├── 2026-05-03_bilstm_setup.md
    └── 2026-05-03_phase2_3_complete.md  ← 이 파일
```

---

## 🔑 주요 설계 결정 (추가)

### 1. Label remapping 전략
원본 (0=Wake, 1=REM, 2=NREM) → binary (0=REM, 1=NREM) + mask
- Wake 위치: dummy label=0, **mask=0** (loss 무시)
- REM 위치: label=0, mask=1
- NREM 위치: label=1, mask=1

### 2. Center mask 구조 (40-to-20 구현)
```
sequence index: 0 1 2 ... 9 10 11 ... 28 29 30 ... 39
center_window:  0 0 0 ... 0  1  1 ...  1  1  0 ...  0
                ↑ head(10) ↑       ↑ tail(10) ↑
                drop          loss        drop
```
Wake mask AND center mask → **REM/NREM 이고 가운데 위치인 epoch만 loss**.

### 3. 학습 시 시퀀스 sampling
- 환자별 num_epochs 가중치 → random pick (긴 환자가 더 자주 뽑힘 — 균일성 보장)
- random start position (0 ~ N-seq_len)
- DataLoader worker 마다 numpy seed 분리 (`worker_init_fn`)

### 4. 평가 시 sliding window aggregation
각 epoch가 등장하는 모든 window의 logits을 평균 → argmax.
- stride=20, seq_len=40 → 가운데 20을 다음 window의 가장자리와 겹치게
- 환자 시작/끝 부근도 마지막 window로 cover (start = N-seq_len 강제 추가)

### 5. Hierarchical 평가 시 Stage 1 모델 import
`models` 이름이 bilstm/와 resnet/pytorch/ 양쪽에 존재 → 충돌.
해결: `importlib.util.spec_from_file_location('resnet_models', ...)` 으로 별칭 로드.

### 6. Stage 1 + Stage 2 결합 로직
```python
final = np.where(s1 == 0, 0,                # Stage1=Wake → 0
        np.where(s2 == 0, 1, 2))            # Stage1=Sleep → Stage2: 0(REM)→1, 1(NREM)→2
```

---

## 🧪 검증 결과 (smoke test)

각 모듈 단독 실행으로 검증 완료:

```bash
$ python3 models/bilstm_classifier.py
Params: 1.32M
Input  shape: (4, 40, 2048)
Output shape: (4, 40, 2)
✓ Forward pass OK

$ python3 utils/losses.py
case 1 (center 20 only): loss = 0.1347
case 2 (random Wake mask): loss = 0.1583
case 3 (all masked): loss = 0.0000 (should be 0)
✓ MaskedFocalLoss tests passed

$ python3 datasets/seq_dataset.py    # 가짜 .npy 5명으로 검증
[train] patients=3, total_epochs=2,850, __len__=71
[train] REM=141 (6.2%) NREM=2,141  Wake(masked)=568
[val] patients=1, windows=42
✓ SoundSequenceDataset tests passed

$ python3 train.py --help            # CLI 정상
$ python3 eval.py --help             # CLI 정상
$ python3 hierarchical_eval.py --help # CLI 정상 (importlib 경유 OK)
```

---

## 🚀 실행 순서 (GPU 환경에서)

### 한 번에 (처음 실행)
```bash
cd /home/yk/Desktop/sleeptech/project1/bilstm
bash scripts/run_all.sh
```

### 단계별 (디버그/재실행)
```bash
# 1. Embedding 추출 (1회만, ~30분 GPU)
bash scripts/1_extract_embeddings.sh

# 2. 학습 (~수시간)
bash scripts/2_train.sh

# 3. 평가
bash scripts/3_eval.sh
```

### 기본 학습 하이퍼파라미터 (`scripts/2_train.sh`)
| 항목 | 값 |
|---|---|
| seq_len / center_window / stride | 40 / 20 / 20 |
| batch_size | 32 |
| num_epochs | 50 (early stopping patience=15) |
| learning_rate | 1e-3 |
| weight_decay | 1e-4 |
| lr_patience | 5 (ReduceLROnPlateau, factor=0.5) |
| grad_clip | 1.0 |
| focal (α, γ) | (0.75, 3.0) |
| 모델 (proj/hidden/layers/dropout) | 256 / 128 / 2 / 0.3 |

---

## 📊 평가 출력 형식

### `eval.py` 출력
- **per-window**: 학습 시 검증과 동일 조건 (가운데 20 epoch만, mask 적용)
- **per-epoch**: 전체 epoch 커버 (sliding window logits aggregation, 더 robust)

각각 balanced_acc, macro_f1, REM/NREM precision/recall/F1, confusion matrix

### `hierarchical_eval.py` 출력
- 3-class (Wake/REM/NREM) accuracy, balanced_acc, macro_f1
- Per-class precision/recall/F1
- 3x3 confusion matrix
- 환자별 accuracy + Stage 1 wake/sleep 비율
- **비교 baseline**: v1 (EOG_CNN hierarchical) 73.7% balanced

---

## 🎯 성공 기준 (목표)

| Metric | v1 (EOG_CNN) | v2 (BiLSTM 소리) 목표 | 의미 |
|---|---|---|---|
| 3-class balanced acc | 73.7% | **75%+** | 기존 대비 동등 이상 |
| 3-class macro F1 | 0.7392 | **0.74+** | |
| REM recall | 63.9% | **65%+** | REM 검출 개선 |
| 의존성 | EOG 채널 필요 | **소리만** | 실용성↑ |

→ 만약 v2가 v1과 비슷하거나 약간 못해도, **EOG 의존성 제거**라는 큰 가치.
→ 더 좋다면 다음 단계로 noise suppression / SpecAugment 추가 검토.

---

## ⏭️ 다음 일정

1. **GPU 환경 준비** (`nvidia-smi` 정상 동작 확인)
2. `scripts/run_all.sh` 실행 → 전체 파이프라인 1회
3. 결과 분석:
   - `eval_test.json`: BiLSTM REM/NREM 단독 성능
   - `hierarchical_3class.json`: 통합 3-class 성능
   - v1과 비교 → 개선/동등/저하 판단
4. (필요 시) Stage B 보강:
   - Noise suppression (`noisereduce` 라이브러리)
   - SpecAugment (학습 시점에서)
   - Embedding 옵션 B (우리 finetuned ResNet22) 비교

---

## 📝 결정 로그 (추가)

| 결정 | 선택 | 이유 |
|---|---|---|
| 학습 metric | val macro F1 (best 기준) | REM 불균형 환경에서 가장 공정 |
| Optimizer | Adam (Hong은 SGD) | 빠른 수렴 + EOG_CNN과 일관성 |
| Grad clipping | 1.0 | LSTM gradient explosion 방지 |
| 평가 방식 | per-window + per-epoch 둘 다 | 학습 조건과 실사용 시나리오 모두 측정 |
| Hierarchical Stage 1 | 기존 `full_ver_2class/best_model.pth` | 가장 검증된 Wake/Sleep 모델 |
| 한 환자 너무 짧으면 | seq_len 미만은 skip | 시작/끝 epoch 부족한 환자 안전 처리 |
