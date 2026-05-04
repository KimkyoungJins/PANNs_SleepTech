# 2026-05-03 — BiLSTM 프로젝트 셋업

## 🎯 목표

소리 데이터만 가지고 **REM vs NREM** 을 구분하는 모델을 만든다.

기존 hierarchical 파이프라인:
- Stage 1 (소리, ResNet22): Wake / Sleep — 잘 됨 (83% acc)
- Stage 2 (EOG, EOG_CNN): REM / NREM — EOG 신호 필요

→ Stage 2를 **소리 기반 BiLSTM**으로 교체해서 EOG 없이 동작하게 만드는 것이 이번 작업.

---

## 📚 배경 (논문 정리)

### [1] Dafna et al. 2018 (Sci Rep) — 전통 ML 접근
- 250명 OSA 환자, 디렉셔널 마이크 1m
- 수작업 67개 feature (호흡 주기성, 들숨/날숨 간격 CV, 체동, 환경음)
- 1-layer NN 앙상블 → **3-class 86.9% acc, REM recall 72.4%, NREM 91.5%**
- 핵심 인사이트: **REM = 호흡 변동성↑ + 체동 거의 없음**, NREM = 규칙적 호흡

### [2] Hong et al. 2022 (Nat Sci Sleep) — SoundSleepNet (end-to-end DL)
- 1154명, 천장 마이크
- **40-to-20 many-to-many** 구조: 40 epoch 입력 → 가운데 20 epoch 예측
- CNN + BiLSTM + Transformer encoder
- 2-step 학습: pretraining(1-to-1) → fine-tune(many-to-many)
- 결과: 4-class 70.3%, 3-class 79.8%, 2-class 89.4%
- **Ablation에서 one-to-one(49.3%) → 40-to-20(70.3%) — 시퀀스가 결정적**
- REM 4-class에서 70%, Light vs Deep 혼동 46%

### [3] Kong et al. 2020 (TASLP) — PANNs
- AudioSet 1.9M으로 사전학습된 audio backbone
- 우리가 쓰는 ResNet22(mAP=0.430), CNN6(mAP=0.343)의 원본
- Fine-tune > freeze (충분한 데이터 시)
- Mixup + SpecAugment로 +5.7%p
- 결론: 우리는 **현재 augmentation 미적용 + 단일 epoch** → 두 군데 손해 보고 있음

### 종합 시사점
1. 단일 epoch 분류는 한계 명확 (REM 변동성은 분 단위 패턴)
2. **시퀀스 모델 도입이 ROI 가장 큼** (Hong ablation 21%p 차이)
3. PANNs ResNet22 backbone 그대로 쓰되 위에 BiLSTM 얹는 게 합리적

---

## 🏗️ 아키텍처 결정

### 전체 파이프라인
```
[Stage 1] 기존 ResNet22 (Wake/Sleep) 유지
[Stage 2] NEW: PANNs ResNet22(frozen) → 2048-d embedding → BiLSTM → REM/NREM
```

### 왜 두 단계로 분리했는가 (Embedding 추출 + BiLSTM 학습)
- ResNet22(63M params)를 frozen으로 쓸 거라 매번 forward 불필요
- 한 번 추출하고 .npy로 캐싱 → **학습 속도 10배↑**, GPU 메모리↓
- 디스크 사용량: 149명 × ~1000 epoch × 2048 × 4byte ≈ **1.2GB** (감당 가능)

### 왜 원본 PANNs 사용 (vs 우리 finetuned ResNet22)
| 항목 | 원본 PANNs | 우리 Wake/Sleep finetuned |
|---|---|---|
| 학습 task | AudioSet 527 클래스 | Wake vs Sleep 2클래스 |
| 표현 풍부도 | 호흡, 코골이, 체동, 환경음 모두 구별 | Wake/Sleep 구별에만 특화 |
| REM/NREM 정보 | **보존됨** (둘 다 sound 카테고리로 학습됨) | 손실 가능성 (둘 다 'Sleep'으로 처리) |

→ **원본 PANNs 채택**. 나중에 비교 실험 가능 (옵션 B로 재추출만 하면 됨).

### BiLSTM 설계 (예정)
```
Input  [B, 40, 2048]                  # 40 epoch = 20분 윈도우
  ↓ Linear(2048 → 256) + Dropout
  ↓ BiLSTM(256, hidden=128, 2-layer, dropout=0.3)
  ↓ Linear(256 → 2)                   # 256 = 128*2 (bidirectional)
Output [B, 40, 2]                     # 모든 epoch 예측
Loss   가운데 20개에만 적용 + Wake epoch mask 처리
```

- Transformer 생략 (149명 데이터 → 과적합 위험)
- 부족하면 Transformer encoder 1-layer 추가
- 40-to-20: Hong 2022 검증된 구조

---

## 📁 디렉토리 구조

```
project1/bilstm/                  ← NEW (이번 작업 전용)
├── extract_embeddings.py         ✓ 작성 완료
├── models/                       (예정: bilstm_classifier.py)
├── datasets/                     (예정: seq_dataset.py)
├── utils/                        (예정: losses.py, metrics.py)
├── scripts/
│   └── 1_extract_embeddings.sh   ✓ 작성 완료
├── embeddings/                   ← .npy 캐시 (~1.2GB)
├── checkpoints/                  ← v1, v2 가중치
├── workspaces/                   ← 학습 결과/로그
└── record/                       ← 개발 일지 (이 파일)
```

`bilstm/` (하이픈 X — Python import 호환성 때문에 underscore도 아닌 그냥 붙여서)

---

## ✅ 오늘 완료한 것

### Phase 1: Embedding 추출 스크립트 작성

**파일**: `bilstm/extract_embeddings.py`

**핵심 동작**:
1. `ResNet22(classes_num=527)` 초기화 → AudioSet pretrained `ResNet22_mAP=0.430.pth` strict 로드
2. `model.eval()` + `requires_grad=False` (frozen)
3. `train.csv + val.csv + test_full.csv` 합쳐서 환자별 모든 라벨 epoch 매핑
4. 환자별 wav를 epoch 순서로 정렬, batch=16으로 forward
5. `output_dict['embedding']` (2048-d, fc_audioset 직전, eval 모드라 dropout=identity) 추출
6. 환자당 3개 파일 저장:
   - `{pid}_emb.npy`: `[N, 2048]` float32
   - `{pid}_labels.npy`: `[N]` int64 (0=W, 1=R, 2=N)
   - `{pid}_meta.json`: split, epoch_ids, label_counts, embedding_source

**Audio params** (resnet/pytorch/main.py와 동일):
- sr=16000, clip=480000, window=512, hop=160, mel=64, fmin=50, fmax=8000

**안전장치**:
- `--skip_existing`: 중간 끊겨도 재시작 가능
- `--limit N`: 디버그용 환자 수 제한
- 환자별 split 일관성 assert (train/val/test 섞이면 에러)
- CUDA 미가용 시 자동 CPU fallback (경고 출력)

**검증**:
- ResNet22 import OK
- 모든 입력 경로 (pretrained, data_dir, csv 3종) 존재 확인
- CLI help 정상

**미실행 사유**: 현재 머신 GPU 미가용 (`nvidia-smi` 실패) → GPU 환경에서 추후 실행

---

## 🚀 다음 단계 (TODO)

### Phase 2: 데이터셋 + 모델 (예정)
- [ ] `extract_embeddings.py` 실제 실행 (5명 디버그 → 149명 전체)
- [ ] 추출 결과 검증 (.npy shape, label 분포)
- [ ] `datasets/seq_dataset.py` — 환자별 40-epoch sequence Dataset
  - random crop (학습), sliding window (평가)
  - Wake mask 처리 (REM/NREM에만 loss)
- [ ] `models/bilstm_classifier.py` — BiLSTM 모델
- [ ] `utils/losses.py` — Focal loss (α=0.75, γ=3.0)

### Phase 3: 학습/평가 (예정)
- [ ] `train.py` — Adam + ReduceLROnPlateau + early stopping(15)
- [ ] `eval.py` — REM/NREM 단독 성능 (balanced acc, macro F1, confusion matrix)
- [ ] `hierarchical_eval.py` — Stage 1(기존 ResNet22) + Stage 2(BiLSTM) 통합

### Phase 4: 비교 실험 (선택)
- [ ] 옵션 B (우리 finetuned ResNet22)로 embedding 재추출
- [ ] BiLSTM 동일 구조로 학습 → A vs B 비교
- [ ] Transformer encoder 추가 ablation

### 목표 지표
- v1 hierarchical (EOG_CNN 기반): 3-class **73.7%** balanced
- 새 v2 (BiLSTM 소리 기반): **75%+ 목표** (+ EOG 의존성 제거 → 실용성↑)

---

## 📝 결정 로그

| 결정 | 선택 | 이유 |
|---|---|---|
| 디렉토리 위치 | `project1/bilstm/` (sibling of resnet/, cnn6/) | 모델별 독립 구조 일관성 |
| 디렉토리 이름 | `bilstm` (하이픈 X) | Python import 호환 |
| Embedding 소스 | 원본 PANNs (옵션 A) | REM/NREM 미세 특징 보존 |
| 추출 방식 | 사전 캐싱 (.npy) | ResNet22 frozen 고정, 속도/메모리 |
| 시퀀스 길이 | 40-to-20 (Hong 2022 그대로) | 검증된 구조, REM 사이클 1개 충분 |
| 모델 시작 구조 | BiLSTM only (Transformer 제외) | 149명 데이터 → 과적합 방지 |
| Loss | Focal (α=0.75, γ=3.0) | EOG_CNN에서 검증됨, REM 불균형 |
