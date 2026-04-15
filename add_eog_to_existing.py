#!/usr/bin/env python3
"""
기존 정제된 환자에게 EOG 데이터만 추가로 생성하는 스크립트.

- 기존 .wav 파일은 건드리지 않음 (마이크 그대로 유지)
- 기존에 이미 처리된 환자들의 EDF를 다시 읽어서
  EOG 채널만 추출 → _eog.npy 저장
- processed.json의 환자 번호 기준으로 원본 EDF 찾음
- 이미 _eog.npy가 있으면 건너뜀 (재실행 안전)

사용법:
  python3 add_eog_to_existing.py

전제:
  - data_for_saving/patient{NN}_{pid}/ 안에 원본 EDF가 있어야 함
  - 또는 new_data/에 EDF가 있어야 함
  - data_for_ai/full_ver/patient{NN}/ 안에 wav 파일이 있어야 함
"""

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import pyedflib
from scipy.signal import resample_poly, butter, filtfilt
from math import gcd

# ─── 설정 (process_sleep_data.py와 동일) ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data")
SAVING_DIR = os.path.join(BASE_DIR, "data_for_saving")
OUTPUT_BASE = os.path.join(BASE_DIR, "data_for_ai")
FULL_DIR = os.path.join(OUTPUT_BASE, "full_ver")
RATIO_DIR = os.path.join(OUTPUT_BASE, "ratio_ver")
MANIFEST_PATH = os.path.join(BASE_DIR, "processed.json")

EPOCH_SEC = 30
EOG_TARGET_SR = 100
EOG_LOC_NAME = "EOG LOC-A2"
EOG_ROC_NAME = "EOG ROC-A2"
EOG_LOWCUT = 0.05
EOG_HIGHCUT = 30.0

NS = "{http://www.respironics.com/PatientStudy.xsd}"

STAGE_MAP = {
    "Wake": 0,
    "NonREM1": 2, "NonREM2": 2, "NonREM3": 2,
    "REM": 1,
}


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {"last_num": 0, "patients": {}}
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    if "patients" not in data:
        patients = data
        last_num = max(patients.values()) if patients else 0
        return {"last_num": last_num, "patients": patients}
    return data


def find_channel_by_name(reader, name):
    labels = reader.getSignalLabels()
    for i, label in enumerate(labels):
        if label.strip() == name:
            return i
    return None


def resample_audio(data, src_sr, tgt_sr):
    if src_sr == tgt_sr:
        return data
    g = gcd(int(src_sr), int(tgt_sr))
    return resample_poly(data, int(tgt_sr) // g, int(src_sr) // g)


def bandpass_filter_eog(data, fs, low=EOG_LOWCUT, high=EOG_HIGHCUT, order=4):
    nyquist = fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    if data.ndim == 1:
        return filtfilt(b, a, data)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def normalize_eog(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    return (data - mean) / std


def parse_rml_stages(rml_path):
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


def find_edf_rml_for_patient(pid, patient_num):
    """환자 번호와 ID로 원본 EDF/RML 찾기.

    우선순위:
      1. data_for_saving/patient{NN}_{pid}/
      2. new_data/ (flat 또는 서브디렉토리)
    """
    patient_name = f"patient{patient_num:02d}"

    # 1. data_for_saving에서 찾기
    saving_dir = os.path.join(SAVING_DIR, f"{patient_name}_{pid}")
    if os.path.isdir(saving_dir):
        rml_files = [f for f in os.listdir(saving_dir) if f.endswith(".rml")]
        edf_files = sorted([f for f in os.listdir(saving_dir) if f.endswith(".edf")])
        if rml_files and edf_files:
            rml_path = os.path.join(saving_dir, rml_files[0])
            edf_paths = [os.path.join(saving_dir, f) for f in edf_files]
            return rml_path, edf_paths

    # 2. new_data에서 flat 구조로 찾기
    if os.path.isdir(NEW_DATA_DIR):
        rml_name = f"{pid}.rml"
        rml_path = os.path.join(NEW_DATA_DIR, rml_name)
        if os.path.exists(rml_path):
            edfs = sorted([
                os.path.join(NEW_DATA_DIR, e)
                for e in os.listdir(NEW_DATA_DIR)
                if e.startswith(pid + "[") and e.endswith(".edf")
            ])
            if edfs:
                return rml_path, edfs

        # 서브디렉토리 구조
        subdir = os.path.join(NEW_DATA_DIR, pid)
        if os.path.isdir(subdir):
            rml_files = [f for f in os.listdir(subdir) if f.endswith(".rml")]
            edf_files = sorted([f for f in os.listdir(subdir) if f.endswith(".edf")])
            if rml_files and edf_files:
                return (os.path.join(subdir, rml_files[0]),
                        [os.path.join(subdir, f) for f in edf_files])

    return None, None


def process_eog_for_patient(pid, patient_num, rml_path, edf_paths):
    """한 환자의 EOG만 추출해서 기존 폴더에 추가."""
    patient_name = f"patient{patient_num:02d}"
    patient_dir = os.path.join(FULL_DIR, patient_name)

    if not os.path.isdir(patient_dir):
        print(f"  [SKIP] {patient_name} 폴더 없음: {patient_dir}")
        return 0, 0

    # 기존 wav 파일 목록
    existing_wavs = sorted([
        f for f in os.listdir(patient_dir)
        if f.endswith(".wav") and f.startswith(patient_name)
    ])

    if not existing_wavs:
        print(f"  [SKIP] {patient_name}: wav 파일 없음")
        return 0, 0

    # RML 파싱
    stages = parse_rml_stages(rml_path)
    if not stages:
        print(f"  [SKIP] {patient_name}: UserStaging 없음")
        return 0, 0

    # EDF 정보 수집
    edf_infos = []
    total_sec = 0
    for edf_path in edf_paths:
        try:
            reader = pyedflib.EdfReader(edf_path)
        except OSError as e:
            print(f"    [SKIP EDF] {os.path.basename(edf_path)}: {e}")
            continue

        eog_loc_idx = find_channel_by_name(reader, EOG_LOC_NAME)
        eog_roc_idx = find_channel_by_name(reader, EOG_ROC_NAME)

        if eog_loc_idx is None or eog_roc_idx is None:
            reader.close()
            continue

        eog_sr = reader.getSampleFrequency(eog_loc_idx)
        dur = reader.file_duration
        edf_infos.append({
            'path': edf_path,
            'eog_loc_idx': eog_loc_idx,
            'eog_roc_idx': eog_roc_idx,
            'eog_sr': eog_sr,
            'dur': dur,
        })
        total_sec += dur
        reader.close()

    if not edf_infos:
        print(f"  [SKIP] {patient_name}: EOG 채널 없음")
        return 0, 0

    epoch_labels = stages_to_epoch_labels(stages, int(total_sec))

    # EDF 단위로 처리
    eog_samples_per_epoch = EOG_TARGET_SR * EPOCH_SEC  # 3000
    elapsed_sec = 0
    saved_count = 0
    skipped_count = 0

    for info in edf_infos:
        edf_path = info['path']
        eog_loc_idx = info['eog_loc_idx']
        eog_roc_idx = info['eog_roc_idx']
        eog_sr = info['eog_sr']
        dur = info['dur']

        edf_start_epoch = elapsed_sec // EPOCH_SEC
        edf_end_epoch = (elapsed_sec + int(dur)) // EPOCH_SEC

        try:
            reader = pyedflib.EdfReader(edf_path)
            raw_loc = reader.readSignal(eog_loc_idx)
            raw_roc = reader.readSignal(eog_roc_idx)
            reader.close()
        except OSError as e:
            print(f"    [SKIP EDF] {os.path.basename(edf_path)}: {e}")
            elapsed_sec += int(dur)
            continue

        raw_eog = np.stack([raw_loc, raw_roc], axis=0)
        del raw_loc, raw_roc

        # 리샘플
        eog_resampled = np.stack([
            resample_audio(raw_eog[0], eog_sr, EOG_TARGET_SR),
            resample_audio(raw_eog[1], eog_sr, EOG_TARGET_SR),
        ], axis=0).astype(np.float32)
        del raw_eog

        # 필터
        eog_filtered = bandpass_filter_eog(eog_resampled, EOG_TARGET_SR)
        del eog_resampled

        # 정규화
        eog_processed = normalize_eog(eog_filtered).astype(np.float32)
        del eog_filtered

        # 에포크 저장
        for i in range(int(edf_start_epoch), min(int(edf_end_epoch), len(epoch_labels))):
            label = epoch_labels[i]
            if label is None:
                continue

            wav_filename = f"{patient_name}_epoch{i+1:04d}.wav"
            eog_filename = f"{patient_name}_epoch{i+1:04d}_eog.npy"

            # wav가 존재하는 에포크만 처리
            if wav_filename not in existing_wavs:
                continue

            # 이미 eog.npy 있으면 건너뜀
            eog_path = os.path.join(patient_dir, eog_filename)
            if os.path.exists(eog_path):
                skipped_count += 1
                continue

            # EOG 에포크 추출
            local_start_sec = i * EPOCH_SEC - elapsed_sec
            eog_start = int(local_start_sec * EOG_TARGET_SR)
            eog_end = eog_start + eog_samples_per_epoch

            if eog_start < 0 or eog_end > eog_processed.shape[1]:
                continue

            eog_chunk = eog_processed[:, eog_start:eog_end]  # [2, 3000]
            np.save(eog_path, eog_chunk)
            saved_count += 1

        del eog_processed
        elapsed_sec += int(dur)

    return saved_count, skipped_count


def copy_eog_to_ratio_ver():
    """ratio_ver에도 EOG 복사 (CSV에 포함된 에포크만)."""
    if not os.path.isdir(RATIO_DIR):
        print("\n[SKIP] ratio_ver 없음")
        return

    print(f"\n{'='*60}")
    print("ratio_ver에 EOG 복사")
    print("="*60)

    copied = 0
    for patient_folder in sorted(os.listdir(RATIO_DIR)):
        patient_path = os.path.join(RATIO_DIR, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        if not patient_folder.startswith("patient"):
            continue

        wav_files = [f for f in os.listdir(patient_path) if f.endswith(".wav")]
        for wav in wav_files:
            eog_name = wav.replace(".wav", "_eog.npy")
            src = os.path.join(FULL_DIR, patient_folder, eog_name)
            dst = os.path.join(patient_path, eog_name)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                copied += 1

    print(f"  ratio_ver에 {copied}개 EOG 파일 복사 완료")


def main():
    print("="*60)
    print("기존 환자 EOG 보충 추출")
    print("="*60)

    manifest = load_manifest()
    processed = manifest["patients"]

    if not processed:
        print("\n처리된 환자가 없습니다. processed.json이 비어있음.")
        return

    print(f"\n총 {len(processed)}명의 환자 처리 이력 확인")

    # 환자 번호 순으로 정렬
    patients_sorted = sorted(processed.items(), key=lambda x: x[1])

    total_saved = 0
    total_skipped = 0
    total_not_found = 0

    for pid, patient_num in patients_sorted:
        # 가짜 legacy 환자는 건너뜀
        if pid.startswith("__legacy"):
            continue

        patient_name = f"patient{patient_num:02d}"
        print(f"\n{'='*60}")
        print(f"{patient_name} ({pid})")

        # 원본 EDF/RML 찾기
        rml_path, edf_paths = find_edf_rml_for_patient(pid, patient_num)
        if rml_path is None or not edf_paths:
            print(f"  [SKIP] 원본 EDF/RML 없음")
            total_not_found += 1
            continue

        print(f"  RML: {os.path.basename(rml_path)}, EDF: {len(edf_paths)}개")

        saved, skipped = process_eog_for_patient(pid, patient_num, rml_path, edf_paths)
        print(f"  저장: {saved}개 (이미 존재: {skipped}개)")
        total_saved += saved
        total_skipped += skipped

    # ratio_ver에도 EOG 복사
    copy_eog_to_ratio_ver()

    print(f"\n{'='*60}")
    print("완료!")
    print(f"  신규 저장: {total_saved}개")
    print(f"  이미 존재: {total_skipped}개 (건너뜀)")
    print(f"  원본 없음: {total_not_found}명")
    print("="*60)


if __name__ == "__main__":
    main()
