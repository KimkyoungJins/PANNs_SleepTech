# 수면 데이터 전처리 파이프라인 (`process_sleep_data.py`)

수면다원검사(PSG) 원본 데이터(EDF + RML)를 AI 학습용 데이터셋으로 변환하는 스크립트.

---

## 1. 전체 파이프라인 흐름

```
new_data/                              process_sleep_data.py
  ├── 00001252-100507.rml          ┐
  ├── 00001252-100507[001].edf     ┤
  ├── 00001252-100507[002].edf     ┤
  ├── ...                          ┤
  ├── 00001414-100507/             ┤     ┌─ data_for_saving/        (원본 백업)
  │   ├── *.rml                    ┤     │
  │   └── *.edf                    ┤ ──▶ ├─ data_for_ai/full_ver/   (전체 데이터)
  └── ...                          ┘     │
                                         └─ data_for_ai/ratio_ver/  (밸런싱 데이터)
```

---

## 2. 최종 출력 디렉토리 구조

### 2.1 `data_for_saving/` — 원본 백업 (환자별)

```
data_for_saving/
├── patient01_00000995-100507/
│   ├── 00000995-100507.rml
│   ├── 00000995-100507[001].edf
│   ├── 00000995-100507[002].edf
│   └── ...
├── patient02_00001014-100507/
│   └── ...
└── patient25_00001480-100507/
    └── ...
```

### 2.2 `data_for_ai/full_ver/` — 전체 데이터 (환자별)

```
data_for_ai/full_ver/
├── patient01/
│   ├── patient01_epoch0001.wav
│   ├── patient01_epoch0002.wav
│   └── ... (해당 환자의 모든 에포크)
├── patient02/
│   └── ...
├── ...
├── patient25/
│   └── ...
├── train.csv          ← 환자 01~15 (60%)
├── val.csv            ← 환자 16~20 (20%)
└── test.csv           ← 환자 21~25 (20%)
```

### 2.3 `data_for_ai/ratio_ver/` — 밸런싱 데이터 (환자별)

```
data_for_ai/ratio_ver/
├── patient01/
│   └── (밸런싱에 선택된 WAV만)
├── patient03/
│   └── ...
├── ...
├── train.csv          ← nrem = rem = wake (동일 수)
├── val.csv            ← nrem = rem = wake (동일 수)
└── test.csv           ← nrem = rem = wake (동일 수)
```

- full_ver에서 가장 적은 클래스 수에 맞춰 나머지를 랜덤 샘플링
- 환자 폴더 구조 그대로 유지 (선택된 파일만 복사)

---

## 3. 원본 데이터 구조 (입력)

### 3.1 디렉토리 레이아웃

`new_data/` 안에 환자 데이터가 **두 가지 형태**로 존재:

| 형태 | 예시 | 설명 |
|------|------|------|
| **Flat** | `new_data/00001252-100507.rml` + `[001].edf ~ [007].edf` | RML과 EDF가 같은 디렉토리에 |
| **서브디렉토리** | `new_data/00001414-100507/*.rml` + `*.edf` | 환자 ID 폴더 안에 모두 포함 |

`find_patients()` 함수가 두 형태를 모두 자동 탐색한다.
파일명에 `[`, `]`가 포함되어 있어 `glob` 대신 `os.listdir()`로 처리.

### 3.2 EDF 파일 (European Data Format)

PSG 장비(Alice 6 LDx)에서 기록한 다채널 생체신호.

| 채널 인덱스 | 라벨 | 샘플레이트 | 설명 |
|------------|------|-----------|------|
| 0 | EEG A1-A2 | 200 Hz | 뇌파 |
| 1~2 | EEG C3-A2, C4-A1 | 200 Hz | 뇌파 |
| 3~4 | EOG LOC/ROC | 200 Hz | 안전도 |
| 5 | EMG Chin | 200 Hz | 근전도 |
| 10 | Snore | 500 Hz | 코골이 센서 |
| **18** | **Mic** | **48,000 Hz** | **수면 소리 — 이것만 추출** |
| 19 | Tracheal | 48,000 Hz | 기관 소리 |
| ... | (기타) | 다양 | SpO2, ECG 등 |

한 환자당 EDF 파일이 여러 개(`[001]`, `[002]`, ...) 존재하며, 각 파일은 약 **1시간(3,600초)** 분량이다.
파일명 순서대로 이어붙이면 전체 수면 녹음이 된다.

### 3.3 RML 파일 (Respironics Markup Language)

XML 기반의 수면 분석 결과 파일.
네임스페이스: `http://www.respironics.com/PatientStudy.xsd`

스크립트가 참조하는 부분은 **`<UserStaging>`** 섹션:

```xml
<UserStaging>
  <NeuroAdultAASMStaging>
    <Stage Type="Wake" Start="0" />
    <Stage Type="NonREM2" Start="240" />
    <Stage Type="REM" Start="600" />
    <Stage Type="NonREM1" Start="870" />
    <Stage Type="NonREM3" Start="1080" />
    ...
  </NeuroAdultAASMStaging>
</UserStaging>
```

- `Type`: 수면 단계 (`Wake`, `NonREM1`, `NonREM2`, `NonREM3`, `REM`)
- `Start`: 녹화 시작 시점 기준 초 단위 오프셋
- 각 Stage = **"이 시점부터 해당 단계 시작"** (다음 Stage까지 유지)

---

## 4. 처리 단계 상세

### 4.1 환자 탐색 (`find_patients`)

```
new_data/ 스캔
  ├── 서브디렉토리 → .rml + .edf 짝 찾기
  └── flat 파일 → {ID}.rml + {ID}[NNN].edf 짝 찾기
→ 환자 ID 오름차순 정렬 → patient01, patient02, ... 번호 부여
```

현재 데이터: 25명

### 4.2 RML 파싱 (`parse_rml_stages`)

```python
NS = "{http://www.respironics.com/PatientStudy.xsd}"

for user_staging in root.iter(NS + "UserStaging"):
    for stage in user_staging.iter(NS + "Stage"):
        → (Start초, 매핑된 라벨) 추출
```

**Stage 매핑:**

| RML 원본 | 변환 결과 |
|----------|----------|
| `Wake` | `wake` |
| `NonREM1` | `nrem` |
| `NonREM2` | `nrem` |
| `NonREM3` | `nrem` |
| `REM` | `rem` |

### 4.3 에포크 라벨 생성 (`stages_to_epoch_labels`)

Stage 전환 목록 → 30초 에포크 단위 라벨 배열:

```
전체 녹음: |----|----|----|----|----|----|----|----|
           0s   30s  60s  90s  120s 150s 180s 210s 240s

Stage:     Wake@0s ──────────────────── NonREM2@240s ────

에포크:    [wake, wake, wake, wake, wake, wake, wake, wake, nrem, ...]
            #1    #2    #3    #4    #5    #6    #7    #8    #9
```

- 각 에포크의 **시작 시점** 기준으로 활성 Stage 결정
- Stage 전환 이전에 아무 Stage 없으면 `None` → 해당 에포크 **스킵**

### 4.4 오디오 추출 및 WAV 저장 (`process_patient`)

```
[EDF 001] + [EDF 002] + ... + [EDF 007]
    │          │                   │
    └──── np.concatenate ──────────┘
              │
              ▼
     48kHz 연속 오디오 (전체 수면 녹음)
              │
              ▼  resample_poly(data, up=1, down=3)
     16kHz 연속 오디오
              │
              ▼  30초 단위로 슬라이싱 (480,000 samples)
     ┌────────┼────────┬────────┐
     │        │        │        │
   epoch1   epoch2   epoch3   ...
     │        │        │        │
     ▼        ▼        ▼        ▼
  normalize (-1.0 ~ 1.0)
     │        │        │        │
     ▼        ▼        ▼        ▼
  full_ver/patient{NN}/patient{NN}_epoch{XXXX}.wav
```

**리샘플링:**

```python
gcd(48000, 16000) = 16000
up   = 16000 / 16000 = 1
down = 48000 / 16000 = 3
# → scipy.signal.resample_poly: polyphase 안티앨리어싱 필터 적용
```

**WAV 파일 스펙:**

| 항목 | 값 |
|------|-----|
| 길이 | 정확히 30초 |
| 샘플레이트 | 16,000 Hz |
| 채널 | 모노 (1ch) |
| 포맷 | WAV (PCM float32) |
| 샘플 수 | 480,000 (= 16000 x 30) |
| 값 범위 | -1.0 ~ 1.0 (에포크 단위 정규화) |

**정규화:** 각 에포크 내에서 독립적으로 `max(abs)` 로 나눔.
환자 간/에포크 간 절대 볼륨 차이 제거 → 모델이 상대적 패턴에 집중.

### 4.5 원본 파일 백업 (`copy_originals`)

```
new_data/00001252-100507.rml         →  data_for_saving/patient01_00001252-100507/
new_data/00001252-100507[001].edf    →    ├── 00001252-100507.rml
new_data/00001252-100507[002].edf    →    ├── 00001252-100507[001].edf
...                                       └── ...
```

`shutil.copy2()` 사용 (메타데이터 보존).

### 4.6 데이터 분할 (`split_data`)

**환자 단위** 6:2:2 분할 (데이터 누수 방지):

```
25명 기준:
  train : patient01 ~ patient15  (15명, 60%)
  val   : patient16 ~ patient20  ( 5명, 20%)
  test  : patient21 ~ patient25  ( 5명, 20%)
```

같은 환자의 에포크가 train과 test에 동시에 들어가지 않음.

### 4.7 클래스 밸런싱 (`create_balanced_version`)

수면 데이터 특성상 NREM이 70~80%를 차지하여 클래스 불균형이 심함.

**방법: Random Undersampling (각 split 독립 수행)**

```
예시 (train):
  원본:               밸런싱 후:
    nrem: 5000         nrem: 300   ← 랜덤 300개 선택
    wake: 1200   →     wake: 300   ← 랜덤 300개 선택
    rem:   300         rem:  300   ← 전부 사용 (최소)
```

- 3개 클래스(nrem, rem, wake)가 **정확히 동일한 수**가 됨
- `random.seed(42)` 고정으로 재현 가능
- 선택된 WAV만 `ratio_ver/patient{NN}/`에 복사

---

## 5. CSV 파일 형식

```csv
filename,label
patient01_epoch0001.wav,wake
patient01_epoch0002.wav,wake
patient01_epoch0003.wav,nrem
patient01_epoch0004.wav,nrem
patient01_epoch0005.wav,rem
...
```

- `filename`: 파일명만 (경로 X) — 환자폴더는 파일명의 `patient{NN}` 부분에서 유추
- `label`: `wake`, `nrem`, `rem` (소문자)
- full_ver과 ratio_ver 모두 동일한 형식

---

## 6. 파일명 규칙

```
patient{환자번호:02d}_epoch{에포크순서:04d}.wav

예: patient03_epoch0142.wav
    → 3번째 환자(ID 정렬 기준)의 142번째 30초 구간
    → 위치: full_ver/patient03/patient03_epoch0142.wav
```

- 환자번호: 01부터 시작, 원본 ID 오름차순 배정
- 에포크순서: 0001부터, 녹화 시작 기준 시간순
- 라벨 없는 에포크는 저장하지 않으므로 번호가 건너뛸 수 있음

---

## 7. 실행 방법

### 7.1 필수 패키지

```bash
pip install pyedflib soundfile scipy numpy
```

### 7.2 실행

```bash
cd /Users/kim_kyoungkun/Desktop/Embeded/Capston2/PANNs/data_all
python3 process_sleep_data.py
```

### 7.3 콘솔 출력 예시

```
============================================================
수면 데이터 전처리 시작
============================================================

환자 25명 발견
  01. 00000995-100507 (EDF 5개)
  02. 00001014-100507 (EDF 4개)
  ...

============================================================
patient01 (00000995-100507)
  RML: 00000995-100507.rml, EDF: 5개
  녹음: 18000초 (5.0h), SR: 48000Hz
  저장: 580개 {'wake': 120, 'nrem': 380, 'rem': 80}

...

============================================================
전체: 14500개 {'wake': 2100, 'nrem': 10200, 'rem': 2200}

============================================================
full_ver CSV 생성
  분할 (환자 단위):
    train : patient [1, 2, ..., 15]
    val   : patient [16, 17, 18, 19, 20]
    test  : patient [21, 22, 23, 24, 25]
    train.csv: 8700개 {'wake': 1260, 'nrem': 6120, 'rem': 1320}
    val.csv: 2900개 ...
    test.csv: 2900개 ...

============================================================
클래스 밸런싱 → ratio_ver/
  train: 각 클래스 → 1260개
    nrem: 6120 → 1260
    rem: 1320 → 1260
    wake: 1260 → 1260
  ...

완료!
```

---

## 8. 주요 설계 결정

| 결정 사항 | 선택 | 이유 |
|----------|------|------|
| Mic 채널 선택 | `Mic` (인덱스 18, 48kHz) | 환경 소리 기록 채널, 음성 대역 충분 |
| 리샘플링 | `scipy.signal.resample_poly` | polyphase 필터로 앨리어싱 방지, 정수비(3:1)에 효율적 |
| 정규화 단위 | 에포크별 독립 | 환자 간 볼륨 차이 제거 |
| 라벨 없는 구간 | 스킵 | UserStaging 시작 전 구간은 신뢰도 없음 |
| NREM 통합 | N1+N2+N3 → nrem | 3-class 분류 (wake/nrem/rem) |
| 밸런싱 방식 | Random Undersampling | 구현 단순, 증강 없이 공정한 평가 |
| 분할 기준 | 환자 단위 6:2:2 | 같은 환자가 train/test에 섞이면 데이터 누수 |
| 환자별 폴더 | full_ver/patient{NN}/ | 환자 단위 관리 용이 |
