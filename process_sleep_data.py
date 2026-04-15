#!/usr/bin/env python3
"""
수면 데이터 전처리 스크립트
- new_data/{환자ID}/ (organize_new_data.py로 정리된 구조) 에서
  EDF(Mic + EOG 채널) + RML(UserStaging) 파싱
- Mic: noisereduce로 정적 배경 소음 제거 (호흡/코골이 보존)
       → 30초 에포크 WAV (16kHz mono) 저장
- EOG: bandpass(0.05~30Hz) + z-score 정규화
       → 30초 에포크 .npy (2채널, 100Hz) 저장
- sleep stage 라벨링 (wake=0, rem=1, nrem=2)
- full_ver: 전체 데이터 (환자별 폴더)
- ratio_ver: 클래스 밸런싱 데이터 (환자별 폴더)

[증분 처리]
- processed.json에 처리 이력(환자ID → 번호) 기록
- 이미 처리된 환자는 건너뜀
- 서버 전송 후 로컬 삭제해도 번호가 이어짐

[원본 파일 백업]
- 이 스크립트는 원본을 복사하지 않음
- 원본은 organize_new_data.py로 정리된 new_data/{환자ID}/ 폴더가
  그대로 백업 역할 (data_for_saving 불필요)
"""

import os
import shutil
import csv
import json
import numpy as np
import xml.etree.ElementTree as ET
import pyedflib
import soundfile as sf
import noisereduce as nr
from scipy.signal import resample_poly, butter, filtfilt
from math import gcd
from collections import Counter, defaultdict
import random

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data")
OUTPUT_BASE = os.path.join(BASE_DIR, "data_for_ai")
FULL_DIR = os.path.join(OUTPUT_BASE, "full_ver")
FULL_2CLASS_DIR = os.path.join(OUTPUT_BASE, "full_ver_2class")
RATIO_DIR = os.path.join(OUTPUT_BASE, "ratio_ver")
RATIO_2CLASS_DIR = os.path.join(OUTPUT_BASE, "ratio_ver_2class")
MANIFEST_PATH = os.path.join(BASE_DIR, "processed.json")

EPOCH_SEC = 30
TARGET_SR = 16000
SOURCE_SR = 48000

# EOG 설정
EOG_TARGET_SR = 100              # 100Hz로 다운샘플 (200Hz 원본)
EOG_LOC_NAME = "EOG LOC-A2"      # 왼쪽 눈 EOG 채널 이름
EOG_ROC_NAME = "EOG ROC-A2"      # 오른쪽 눈 EOG 채널 이름
EOG_LOWCUT = 0.05                # 밴드패스 필터 하한 (DC 드리프트 제거)
EOG_HIGHCUT = 30.0               # 밴드패스 필터 상한 (고주파 노이즈 제거)

NS = "{http://www.respironics.com/PatientStudy.xsd}"

# 라벨 매핑: wake=0, rem=1, nrem=2
STAGE_MAP = {
    "Wake": 0,
    "NonREM1": 2,
    "NonREM2": 2,
    "NonREM3": 2,
    "REM": 1,
}

LABEL_NAMES = {0: "wake", 1: "rem", 2: "nrem"}

random.seed(42)


# ──────────────────────────────────────────────
# 매니페스트 (처리 이력 관리)
# ──────────────────────────────────────────────
def load_manifest():
    """
    processed.json 로드.
    형식: {"last_num": 25, "patients": {"00001486-100507": 1, ...}}
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            data = json.load(f)
        # 구버전 호환 (dict만 있는 경우)
        if "patients" not in data:
            patients = data
            last_num = max(patients.values()) if patients else 0
            return {"last_num": last_num, "patients": patients}
        return data
    return {"last_num": 0, "patients": {}}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# 노이즈 제거
# ──────────────────────────────────────────────
def denoise_epoch(chunk, sr):
    """
    정적 배경 소음 제거 (noisereduce, stationary mode).
    - prop_decrease=0.5: 배경 소음 50%만 줄임 (호흡/코골이 패턴 보존)
    """
    return nr.reduce_noise(
        y=chunk,
        sr=sr,
        prop_decrease=0.5,
        stationary=True,
    )


# ──────────────────────────────────────────────
# 환자 탐색
# ──────────────────────────────────────────────
def find_patients(data_dir):
    """
    new_data/ 에서 환자 목록 탐색 (서브디렉토리 + flat 구조 모두 지원).
    반환: [(patient_id, rml_path, [edf_paths_sorted]), ...] (ID 정렬)
    """
    patients = {}

    # 서브디렉토리 형태
    for entry in os.listdir(data_dir):
        full = os.path.join(data_dir, entry)
        if os.path.isdir(full):
            rml = None
            edfs = []
            for f in os.listdir(full):
                if f.endswith(".rml"):
                    rml = os.path.join(full, f)
                elif f.endswith(".edf"):
                    edfs.append(os.path.join(full, f))
            if rml and edfs:
                patients[entry] = (rml, sorted(edfs))

    # flat 형태 (new_data/ 직하)
    for f in os.listdir(data_dir):
        if not f.endswith(".rml"):
            continue
        pid = f.replace(".rml", "")
        if pid in patients:
            continue
        rml_path = os.path.join(data_dir, f)
        edfs = [
            os.path.join(data_dir, e)
            for e in os.listdir(data_dir)
            if e.startswith(pid + "[") and e.endswith(".edf")
        ]
        if edfs:
            patients[pid] = (rml_path, sorted(edfs))

    return [(pid, *patients[pid]) for pid in sorted(patients)]


# ──────────────────────────────────────────────
# RML 파싱
# ──────────────────────────────────────────────
def parse_rml_stages(rml_path):
    """UserStaging 에서 Stage 목록 파싱 → [(start_sec, label), ...] 정렬"""
    tree = ET.parse(rml_path)
    root = tree.getroot()
    stages = []
    for us in root.iter(NS + "UserStaging"):
        for s in us.iter(NS + "Stage"):
            stype = s.get("Type")
            start = s.get("Start")
            if stype and start is not None:
                label = STAGE_MAP.get(stype)
                if label is not None:
                    stages.append((int(start), label))
    stages.sort(key=lambda x: x[0])
    return stages


def stages_to_epoch_labels(stages, total_duration_sec):
    """Stage 전환 목록 → 30초 에포크별 라벨 리스트"""
    n_epochs = total_duration_sec // EPOCH_SEC
    labels = [None] * n_epochs
    for idx in range(n_epochs):
        t = idx * EPOCH_SEC
        cur = None
        for start, label in stages:
            if start <= t:
                cur = label
            else:
                break
        labels[idx] = cur
    return labels


# ──────────────────────────────────────────────
# EDF 오디오 처리
# ──────────────────────────────────────────────
def find_mic_channel(reader):
    """Mic 채널 인덱스 찾기 (fallback: 8kHz 이상 첫 채널)"""
    labels = reader.getSignalLabels()
    for i, l in enumerate(labels):
        if l.strip().lower() == "mic":
            return i
    for i in range(reader.signals_in_file):
        if reader.getSampleFrequency(i) >= 8000:
            return i
    return None


def resample_audio(data, src_sr, tgt_sr):
    if src_sr == tgt_sr:
        return data
    g = gcd(int(src_sr), int(tgt_sr))
    return resample_poly(data, int(tgt_sr) // g, int(src_sr) // g)


def normalize_audio(data):
    mx = np.max(np.abs(data))
    return data / mx if mx > 0 else data


# ──────────────────────────────────────────────
# EOG 처리 (신규 추가)
# ──────────────────────────────────────────────
def find_channel_by_name(reader, name):
    """EDF에서 이름으로 채널 인덱스 찾기"""
    labels = reader.getSignalLabels()
    for i, label in enumerate(labels):
        if label.strip() == name:
            return i
    return None


def bandpass_filter_eog(data, fs, low=EOG_LOWCUT, high=EOG_HIGHCUT, order=4):
    """EOG 밴드패스 필터 (0.05~30Hz).
    - 0.05Hz 이하: DC 드리프트 제거 (전극 이동, 땀)
    - 30Hz 이상: 고주파 노이즈 제거 (근전도, 전기 간섭)
    data: [C, N] 또는 [N]
    """
    nyquist = fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    if data.ndim == 1:
        return filtfilt(b, a, data)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def normalize_eog(data):
    """EOG Z-score 정규화 (채널별, 파일 전체 기준).
    환자/녹음 간 EOG 진폭 차이를 제거.
    data: [C, N]
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    return (data - mean) / std


# ──────────────────────────────────────────────
# 환자 1명 처리
# ──────────────────────────────────────────────
def process_patient(pid, rml_path, edf_paths, patient_num, output_dir):
    """
    EDF Mic + EOG 추출 → 노이즈 제거/필터링 → 30초 에포크 저장.
    - Mic: 48kHz → 16kHz, noisereduce, .wav 저장
    - EOG: 200Hz → 100Hz, bandpass 0.05~30Hz, z-score, .npy 저장 (2채널)

    메모리 절약을 위해 EDF 파일 단위(~1시간)로 나누어 처리.
    EOG 채널이 없는 EDF는 Mic만 저장 (기존 동작 유지).
    반환: [(filename, label), ...]
    """
    patient_name = f"patient{patient_num:02d}"
    print(f"\n{'='*60}")
    print(f"{patient_name} ({pid})")
    print(f"  RML: {os.path.basename(rml_path)}, EDF: {len(edf_paths)}개")

    # RML 파싱
    stages = parse_rml_stages(rml_path)
    if not stages:
        print("  [SKIP] UserStaging 없음")
        return []

    # 전체 녹음 길이 먼저 계산 (메모리 로드 없이)
    edf_infos = []
    total_sec = 0
    eog_available_any = False
    for edf_path in edf_paths:
        try:
            reader = pyedflib.EdfReader(edf_path)
        except OSError as e:
            print(f"  [SKIP] {os.path.basename(edf_path)}: 손상된 파일 ({e})")
            continue
        mic_idx = find_mic_channel(reader)
        if mic_idx is None:
            print(f"  [SKIP] {os.path.basename(edf_path)}: Mic 채널 없음")
            reader.close()
            continue

        mic_sr = reader.getSampleFrequency(mic_idx)
        dur = reader.file_duration

        # EOG 채널 탐색 (선택적, 없어도 마이크는 처리)
        eog_loc_idx = find_channel_by_name(reader, EOG_LOC_NAME)
        eog_roc_idx = find_channel_by_name(reader, EOG_ROC_NAME)
        eog_sr = None
        if eog_loc_idx is not None and eog_roc_idx is not None:
            eog_sr = reader.getSampleFrequency(eog_loc_idx)
            eog_available_any = True

        edf_infos.append({
            'path': edf_path,
            'mic_idx': mic_idx,
            'mic_sr': mic_sr,
            'eog_loc_idx': eog_loc_idx,
            'eog_roc_idx': eog_roc_idx,
            'eog_sr': eog_sr,
            'dur': dur,
        })
        total_sec += dur
        reader.close()

    if not edf_infos:
        print("  [SKIP] 오디오 없음")
        return []

    src_sr = edf_infos[0]['mic_sr']
    eog_status = "EOG 있음" if eog_available_any else "EOG 없음 (Mic만 저장)"
    print(f"  녹음: {total_sec:.0f}초 ({total_sec/3600:.1f}h), Mic SR: {int(src_sr)}Hz, {eog_status}")

    # 전체 에포크 라벨 생성
    epoch_labels = stages_to_epoch_labels(stages, int(total_sec))

    # 환자 폴더 생성
    patient_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_dir, exist_ok=True)

    # EDF 파일 단위로 처리 (메모리 절약: ~1시간분만 메모리에)
    results = []
    mic_samples_per_epoch = TARGET_SR * EPOCH_SEC       # 480,000
    eog_samples_per_epoch = EOG_TARGET_SR * EPOCH_SEC   # 3,000
    elapsed_sec = 0

    for info in edf_infos:
        edf_path = info['path']
        mic_idx = info['mic_idx']
        mic_sr = info['mic_sr']
        eog_loc_idx = info['eog_loc_idx']
        eog_roc_idx = info['eog_roc_idx']
        eog_sr = info['eog_sr']
        dur = info['dur']

        edf_start_epoch = elapsed_sec // EPOCH_SEC
        edf_end_epoch = (elapsed_sec + int(dur)) // EPOCH_SEC

        # EDF 1개 읽기 + 리샘플링
        try:
            reader = pyedflib.EdfReader(edf_path)
            raw_mic = reader.readSignal(mic_idx)

            # EOG 읽기 (있으면)
            raw_eog = None
            if eog_loc_idx is not None and eog_roc_idx is not None:
                raw_loc = reader.readSignal(eog_loc_idx)
                raw_roc = reader.readSignal(eog_roc_idx)
                raw_eog = np.stack([raw_loc, raw_roc], axis=0)  # [2, N]
                del raw_loc, raw_roc

            reader.close()
        except OSError as e:
            print(f"  [SKIP] {os.path.basename(edf_path)}: 읽기 실패 ({e})")
            elapsed_sec += int(dur)
            continue

        # Mic 리샘플
        audio_16k = resample_audio(raw_mic, mic_sr, TARGET_SR).astype(np.float32)
        del raw_mic

        # EOG 처리 (리샘플 → 필터 → 정규화)
        eog_processed = None
        if raw_eog is not None:
            eog_resampled = np.stack([
                resample_audio(raw_eog[0], eog_sr, EOG_TARGET_SR),
                resample_audio(raw_eog[1], eog_sr, EOG_TARGET_SR),
            ], axis=0).astype(np.float32)
            del raw_eog

            # 밴드패스 필터 (0.05~30Hz)
            eog_filtered = bandpass_filter_eog(eog_resampled, EOG_TARGET_SR)
            del eog_resampled

            # Z-score 정규화 (파일 단위)
            eog_processed = normalize_eog(eog_filtered).astype(np.float32)
            del eog_filtered

        # 이 EDF 내 에포크 처리
        for i in range(int(edf_start_epoch), min(int(edf_end_epoch), len(epoch_labels))):
            label = epoch_labels[i]
            if label is None:
                continue

            # === Mic 에포크 경계 계산 ===
            local_start_sec = i * EPOCH_SEC - elapsed_sec
            mic_start = int(local_start_sec * TARGET_SR)
            mic_end = mic_start + mic_samples_per_epoch

            if mic_start < 0 or mic_end > len(audio_16k):
                continue

            # === EOG 에포크 경계 계산 (있을 때) ===
            eog_chunk = None
            if eog_processed is not None:
                eog_start = int(local_start_sec * EOG_TARGET_SR)
                eog_end = eog_start + eog_samples_per_epoch

                if eog_start < 0 or eog_end > eog_processed.shape[1]:
                    # EOG 경계 벗어나면 이 에포크 스킵 (쌍 일관성)
                    continue

                eog_chunk = eog_processed[:, eog_start:eog_end]  # [2, 3000]

                # EOG 유효성 검증 (NaN, Inf, shape)
                if (eog_chunk.shape != (2, eog_samples_per_epoch) or
                    not np.all(np.isfinite(eog_chunk))):
                    continue

            # === 원자적 저장 (wav + eog 쌍) ===
            base_name = f"{patient_name}_epoch{i+1:04d}"
            wav_filename = f"{base_name}.wav"
            wav_path = os.path.join(patient_dir, wav_filename)
            eog_path = os.path.join(patient_dir, f"{base_name}_eog.npy")

            # 재처리 시 이미 쌍이 완성된 에포크는 건너뜀 (resume 안전)
            wav_exists = os.path.exists(wav_path)
            eog_exists = (eog_chunk is None) or os.path.exists(eog_path)
            if wav_exists and eog_exists:
                results.append((wav_filename, label))
                continue

            try:
                # Mic 전처리
                mic_chunk = audio_16k[mic_start:mic_end]
                mic_chunk = denoise_epoch(mic_chunk, TARGET_SR)
                mic_chunk = normalize_audio(mic_chunk.astype(np.float32))

                # 둘 다 저장 (실패 시 둘 다 롤백)
                sf.write(wav_path, mic_chunk, TARGET_SR)
                if eog_chunk is not None:
                    np.save(eog_path, eog_chunk)

            except Exception as e:
                print(f"  [ERROR] epoch{i+1:04d} 저장 실패: {e}")
                # 둘 다 삭제 (쌍 일관성 유지)
                for p in (wav_path, eog_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                continue

            results.append((wav_filename, label))

        del audio_16k
        if eog_processed is not None:
            del eog_processed
        elapsed_sec += int(dur)

    counter = Counter(r[1] for r in results)
    named = {LABEL_NAMES.get(k, k): v for k, v in sorted(counter.items())}
    print(f"  저장: {len(results)}개 {named}")
    return results


# ──────────────────────────────────────────────
# CSV 작성 + 데이터 분할
# ──────────────────────────────────────────────
def write_csv(filepath, data):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(data)


def label_count_str(data):
    """라벨 카운트를 이름으로 표시"""
    counts = Counter(r[1] for r in data)
    return {LABEL_NAMES.get(k, k): v for k, v in sorted(counts.items())}


def balance_data(data):
    """클래스별 개수를 최소 클래스에 맞춰 균형 맞추기"""
    by_label = defaultdict(list)
    for item in data:
        by_label[item[1]].append(item)

    if not by_label:
        return data

    min_count = min(len(v) for v in by_label.values())
    balanced = []
    for label in sorted(by_label.keys()):
        sampled = random.sample(by_label[label], min_count)
        balanced.extend(sampled)

    balanced.sort(key=lambda x: x[0])
    return balanced


def assign_split_for_patient(pnum, has_wake):
    """환자 번호 기반 결정론적 split 할당.

    같은 환자 번호는 항상 같은 split으로 배정되므로
    데이터 추가 시에도 기존 분할이 유지됨 (Data Leakage 방지).

    wake 환자와 non-wake 환자를 별도로 균등 배분:
      - train: 60% (pnum % 10 ∈ {0,1,2,3,4,5})
      - val:   20% (pnum % 10 ∈ {6,7})
      - test:  20% (pnum % 10 ∈ {8,9})
    """
    m = pnum % 10
    if m <= 5:
        return "train"
    elif m <= 7:
        return "val"
    else:
        return "test"


def split_data(all_results):
    """
    환자 단위 6:2:2 분할 (결정론적).
    환자 번호 기반 split 할당으로 데이터 추가 시에도 안정적.
    test는 항상 클래스 균형 맞춤.
    반환: (train_data, val_data, test_data, test_data_full)
    """
    by_patient = defaultdict(list)
    for filename, label in all_results:
        pnum = int(filename.split("_")[0].replace("patient", ""))
        by_patient[pnum].append((filename, label))

    # 결정론적 분할 (환자 번호 기반)
    train_p = []
    val_p = []
    test_p = []

    for pnum in sorted(by_patient.keys()):
        labels = set(lb for _, lb in by_patient[pnum])
        has_wake = 0 in labels

        split = assign_split_for_patient(pnum, has_wake)
        if split == "train":
            train_p.append(pnum)
        elif split == "val":
            val_p.append(pnum)
        else:
            test_p.append(pnum)

    train_p.sort()
    val_p.sort()
    test_p.sort()

    print(f"\n  총 {len(by_patient)}명 환자 결정론적 분할:")
    print(f"    train: {len(train_p)}명 (60%)")
    print(f"    val:   {len(val_p)}명 (20%)")
    print(f"    test:  {len(test_p)}명 (20%)")

    def collect(pnums):
        out = []
        for p in pnums:
            out.extend(by_patient[p])
        return out

    train_d, val_d = collect(train_p), collect(val_p)
    test_d_full = collect(test_p)

    # test는 CNN용은 클래스 균형, LSTM용은 원본 유지
    test_d = balance_data(test_d_full)

    print(f"\n  분할 (환자 단위):")
    print(f"    train : patient {train_p}")
    print(f"    val   : patient {val_p}")
    print(f"    test  : patient {test_p}")
    print(f"    train.csv: {len(train_d)}개 {label_count_str(train_d)}")
    print(f"    val.csv: {len(val_d)}개 {label_count_str(val_d)}")
    print(f"    test.csv (원본): {len(test_d_full)}개 {label_count_str(test_d_full)}")
    print(f"    test.csv (균형): {len(test_d)}개 {label_count_str(test_d)}")

    return train_d, val_d, test_d, test_d_full


# ──────────────────────────────────────────────
# 2-Class 변환 (wake=0, sleep=1)
# ──────────────────────────────────────────────
LABEL_MAP_2CLASS = {0: 0, 1: 1, 2: 1}  # wake→0, rem→1(sleep), nrem→1(sleep)
LABEL_NAMES_2CLASS = {0: "wake", 1: "sleep"}


def convert_to_2class(data):
    """3-Class 라벨(0,1,2)을 2-Class(0,1)로 변환"""
    return [(fn, LABEL_MAP_2CLASS[lb]) for fn, lb in data]


def create_2class_version(train_data, val_data, test_data, src_dir, dst_dir):
    """2-Class 버전 생성: CSV만 생성 (WAV는 src_dir을 심볼릭 링크)"""
    print(f"\n{'='*60}")
    print(f"2-Class 버전 생성 → {os.path.basename(dst_dir)}/")

    os.makedirs(dst_dir, exist_ok=True)

    # WAV 파일은 원본 폴더의 심볼릭 링크로 연결 (용량 절약)
    for entry in os.listdir(src_dir):
        src_path = os.path.join(src_dir, entry)
        dst_path = os.path.join(dst_dir, entry)
        if os.path.isdir(src_path) and entry.startswith("patient"):
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        data_2class = convert_to_2class(split_data)

        # test는 2-Class 기준으로 균형 맞춤 (wake:sleep = 1:1)
        # CNN용 (개별 에포크 평가)
        if split_name == "test":
            data_2class_balanced = balance_data(data_2class)
        else:
            data_2class_balanced = data_2class

        write_csv(os.path.join(dst_dir, f"{split_name}.csv"), data_2class_balanced)

        counts = {LABEL_NAMES_2CLASS.get(k, k): v for k, v in sorted(Counter(r[1] for r in data_2class_balanced).items())}
        print(f"  {split_name}.csv: {len(data_2class_balanced)}개 {counts}")

    # LSTM용 test CSV: 연속성 보존 (balance_data 미적용)
    test_2class_full = convert_to_2class(test_data)
    write_csv(os.path.join(dst_dir, "test_lstm.csv"), test_2class_full)
    counts_lstm = {LABEL_NAMES_2CLASS.get(k, k): v for k, v in sorted(Counter(r[1] for r in test_2class_full).items())}
    print(f"  test_lstm.csv: {len(test_2class_full)}개 {counts_lstm} (LSTM용, 연속성 보존)")


# ──────────────────────────────────────────────
# 클래스 밸런싱 (3-Class)
# ──────────────────────────────────────────────
def create_balanced_version(train_data, val_data, test_data, test_data_full, full_dir, ratio_dir):
    """
    각 split에서 rem/nrem/wake 비율을 최소 클래스에 맞춰 동일하게 만든다.
    선택된 WAV를 ratio_ver/ 에 환자별 폴더로 복사.
    test_data_full: LSTM용 (연속성 보존, balance_data 미적용)
    """
    print(f"\n{'='*60}")
    print("클래스 밸런싱 → ratio_ver/")

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        by_label = defaultdict(list)
        for filename, label in split_data:
            by_label[label].append((filename, label))

        if not by_label:
            continue

        min_count = min(len(v) for v in by_label.values())
        print(f"\n  {split_name}: 각 클래스 → {min_count}개")

        balanced = []
        for label, items in sorted(by_label.items()):
            sampled = random.sample(items, min_count)
            balanced.extend(sampled)
            print(f"    {LABEL_NAMES.get(label, label)}: {len(items)} → {min_count}")

        balanced.sort(key=lambda x: x[0])

        # WAV + EOG .npy 복사 (환자별 폴더 유지)
        for filename, label in balanced:
            patient_folder = filename.split("_")[0]
            dst_dir = os.path.join(ratio_dir, patient_folder)
            os.makedirs(dst_dir, exist_ok=True)

            # 1) .wav 복사
            src = os.path.join(full_dir, patient_folder, filename)
            dst = os.path.join(dst_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

            # 2) _eog.npy 복사 (있으면)
            eog_name = filename.replace(".wav", "_eog.npy")
            eog_src = os.path.join(full_dir, patient_folder, eog_name)
            eog_dst = os.path.join(dst_dir, eog_name)
            if os.path.exists(eog_src) and not os.path.exists(eog_dst):
                shutil.copy2(eog_src, eog_dst)

        write_csv(os.path.join(ratio_dir, f"{split_name}.csv"), balanced)

    # LSTM용 test CSV: 연속성 보존 (balance_data 미적용)
    write_csv(os.path.join(ratio_dir, "test_lstm.csv"), test_data_full)
    named_lstm = {LABEL_NAMES.get(k, k): v for k, v in sorted(Counter(r[1] for r in test_data_full).items())}
    print(f"\n  test_lstm.csv: {len(test_data_full)}개 {named_lstm} (LSTM용, 연속성 보존)")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("수면 데이터 전처리 시작")
    print("=" * 60)

    # 매니페스트 로드
    manifest = load_manifest()
    last_num = manifest["last_num"]
    processed = manifest["patients"]

    if processed:
        print(f"\n처리 이력: {len(processed)}명 (마지막 번호: patient{last_num:02d})")

    # 환자 탐색
    all_patients = find_patients(NEW_DATA_DIR)
    print(f"\nnew_data/에서 환자 {len(all_patients)}명 발견")

    # 이미 처리된 환자 필터링
    new_patients = [(pid, rml, edfs) for pid, rml, edfs in all_patients
                    if pid not in processed]
    skipped = [(pid, rml, edfs) for pid, rml, edfs in all_patients
               if pid in processed]

    if skipped:
        print(f"\n이미 처리됨 (건너뜀): {len(skipped)}명")
        for pid, _, _ in skipped:
            print(f"  - {pid} (patient{processed[pid]:02d})")

    if not new_patients:
        print("\n새로 처리할 환자가 없습니다!")
        return

    print(f"\n신규 환자: {len(new_patients)}명")
    for pid, rml, edfs in new_patients:
        print(f"  - {pid} (EDF {len(edfs)}개)")

    # 다음 번호 결정
    next_num = last_num + 1
    print(f"\n번호 시작: patient{next_num:02d}")

    # full_ver: 신규 환자만 처리
    os.makedirs(FULL_DIR, exist_ok=True)
    new_results = []
    for i, (pid, rml, edfs) in enumerate(new_patients):
        num = next_num + i
        try:
            results = process_patient(pid, rml, edfs, num, FULL_DIR)
        except KeyboardInterrupt:
            print(f"\n[사용자 중단] patient{num:02d} ({pid}) 처리 중 멈춤")
            print("이미 처리 완료된 환자는 매니페스트에 저장되었습니다.")
            print("재실행 시 처리된 환자부터 이어서 진행합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] patient{num:02d} ({pid}) 처리 실패: {e}")
            print("  → 해당 환자 건너뜀, 매니페스트에 등록 안 함")
            print("  → 원인 해결 후 재실행하면 이 환자부터 다시 시도")
            continue

        if results:
            new_results.extend(results)
            # 매니페스트에 즉시 추가 + 저장 (환자 단위 원자성)
            processed[pid] = num
            manifest["last_num"] = num
            manifest["patients"] = processed
            save_manifest(manifest)
        else:
            print(f"  [SKIP] patient{num:02d} ({pid}): 유효한 에포크 없음")

    print(f"\n{'='*60}")
    named = {LABEL_NAMES.get(k, k): v for k, v in sorted(Counter(r[1] for r in new_results).items())}
    print(f"신규 처리: {len(new_results)}개 {named}")

    # CSV는 전체 환자 기반으로 재생성 (기존 + 신규)
    # 기존 CSV에서 read → 신규와 합침 → 재분할
    all_results = list(new_results)
    for split_name in ("train", "val", "test", "test_lstm"):
        csv_path = os.path.join(FULL_DIR, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                fn, lb = row[0], int(row[1])
                # test.csv는 균형 맞춤이라 원본의 일부만 들어있으므로
                # test_lstm.csv로 복원 (전체 데이터)
                if split_name in ("train", "val", "test_lstm"):
                    all_results.append((fn, lb))

    # 중복 제거 (파일명 기준)
    seen = set()
    deduped = []
    for fn, lb in all_results:
        if fn not in seen:
            seen.add(fn)
            deduped.append((fn, lb))
    all_results = deduped

    if not all_results:
        print("처리된 데이터 없음!")
        return

    print(f"전체 (기존+신규): {len(all_results)}개")

    # CSV 재생성 (전체 기준)
    print(f"\n{'='*60}")
    print("full_ver CSV 재생성 (전체 환자)")
    train_d, val_d, test_d, test_d_full = split_data(all_results)
    write_csv(os.path.join(FULL_DIR, "train.csv"), train_d)
    write_csv(os.path.join(FULL_DIR, "val.csv"), val_d)
    write_csv(os.path.join(FULL_DIR, "test.csv"), test_d)
    # LSTM용 test CSV: 연속성 보존 (balance_data 미적용)
    write_csv(os.path.join(FULL_DIR, "test_lstm.csv"), test_d_full)
    print(f"  test_lstm.csv: {len(test_d_full)}개 {label_count_str(test_d_full)} (LSTM용, 연속성 보존)")

    # ratio_ver (3-Class 균형)
    if os.path.exists(RATIO_DIR):
        shutil.rmtree(RATIO_DIR)
    os.makedirs(RATIO_DIR, exist_ok=True)
    create_balanced_version(train_d, val_d, test_d, test_d_full, FULL_DIR, RATIO_DIR)

    # full_ver_2class (2-Class, test 균형)
    if os.path.exists(FULL_2CLASS_DIR):
        shutil.rmtree(FULL_2CLASS_DIR)
    create_2class_version(train_d, val_d, test_d_full, FULL_DIR, FULL_2CLASS_DIR)

    # ratio_ver_2class (2-Class 균형)
    if os.path.exists(RATIO_2CLASS_DIR):
        shutil.rmtree(RATIO_2CLASS_DIR)
    # ratio_ver의 CSV를 읽어서 2-Class로 변환
    ratio_train = []
    ratio_val = []
    ratio_test = []
    for split_name, split_list in [("train", ratio_train), ("val", ratio_val), ("test", ratio_test)]:
        csv_path = os.path.join(RATIO_DIR, f"{split_name}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    split_list.append((row[0], int(row[1])))
    create_2class_version(ratio_train, ratio_val, ratio_test, RATIO_DIR, RATIO_2CLASS_DIR)

    print(f"\n{'='*60}")
    print("완료!")
    print(f"  full_ver (3class):        {FULL_DIR}")
    print(f"  full_ver_2class:          {FULL_2CLASS_DIR}")
    print(f"  ratio_ver (3class):       {RATIO_DIR}")
    print(f"  ratio_ver_2class:         {RATIO_2CLASS_DIR}")
    print(f"  매니페스트:               {MANIFEST_PATH}")
    print(f"  총 누적 환자:             {manifest['last_num']}명")
    print("=" * 60)


if __name__ == "__main__":
    main()
