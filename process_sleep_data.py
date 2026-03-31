#!/usr/bin/env python3
"""
수면 데이터 전처리 스크립트
- new_data/에서 EDF(Mic 채널) + RML(UserStaging) 파싱
- noisereduce로 정적 배경 소음 제거 (호흡/코골이 보존)
- 30초 에포크 단위 WAV(16kHz mono) 추출
- sleep stage 라벨링 (wake, nrem, rem)
- full_ver: 전체 데이터 (환자별 폴더)
- ratio_ver: 클래스 밸런싱 데이터 (환자별 폴더)
- data_for_saving: 원본 파일 환자별 정리

[증분 처리]
- processed.json에 처리 이력(환자ID → 번호) 기록
- 이미 처리된 환자는 건너뜀
- 서버 전송 후 로컬 삭제해도 번호가 이어짐
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
from scipy.signal import resample_poly
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
SAVING_DIR = os.path.join(BASE_DIR, "data_for_saving")
MANIFEST_PATH = os.path.join(BASE_DIR, "processed.json")

EPOCH_SEC = 30
TARGET_SR = 16000
SOURCE_SR = 48000

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
                if label:
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
# 환자 1명 처리
# ──────────────────────────────────────────────
def process_patient(pid, rml_path, edf_paths, patient_num, output_dir):
    """
    EDF Mic 추출 → 노이즈 제거 → 30초 WAV + 라벨.
    메모리 절약을 위해 EDF 파일 단위(~1시간)로 나누어 처리.
    WAV는 output_dir/patient{NN}/ 에 직접 저장.
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
    for edf_path in edf_paths:
        reader = pyedflib.EdfReader(edf_path)
        mic_idx = find_mic_channel(reader)
        if mic_idx is None:
            print(f"  [SKIP] {os.path.basename(edf_path)}: Mic 채널 없음")
            reader.close()
            continue
        sr = reader.getSampleFrequency(mic_idx)
        dur = reader.file_duration
        edf_infos.append((edf_path, mic_idx, sr, dur))
        total_sec += dur
        reader.close()

    if not edf_infos:
        print("  [SKIP] 오디오 없음")
        return []

    src_sr = edf_infos[0][2]
    print(f"  녹음: {total_sec:.0f}초 ({total_sec/3600:.1f}h), SR: {int(src_sr)}Hz")

    # 전체 에포크 라벨 생성
    epoch_labels = stages_to_epoch_labels(stages, int(total_sec))

    # 환자 폴더 생성
    patient_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_dir, exist_ok=True)

    # EDF 파일 단위로 처리 (메모리 절약: ~1시간분만 메모리에)
    results = []
    samples_per_epoch = TARGET_SR * EPOCH_SEC  # 480,000
    elapsed_sec = 0

    for edf_path, mic_idx, sr, dur in edf_infos:
        edf_start_epoch = elapsed_sec // EPOCH_SEC
        edf_end_epoch = (elapsed_sec + int(dur)) // EPOCH_SEC

        # EDF 1개 읽기 + 리샘플링
        reader = pyedflib.EdfReader(edf_path)
        raw = reader.readSignal(mic_idx)
        reader.close()

        audio_16k = resample_audio(raw, src_sr, TARGET_SR).astype(np.float32)
        del raw

        # 이 EDF 내 에포크 처리
        for i in range(int(edf_start_epoch), min(int(edf_end_epoch), len(epoch_labels))):
            label = epoch_labels[i]
            if label is None:
                continue

            local_start_sec = i * EPOCH_SEC - elapsed_sec
            local_start_sample = int(local_start_sec * TARGET_SR)
            local_end_sample = local_start_sample + samples_per_epoch

            if local_start_sample < 0 or local_end_sample > len(audio_16k):
                continue

            chunk = audio_16k[local_start_sample:local_end_sample]

            # 정적 배경 소음 제거
            chunk = denoise_epoch(chunk, TARGET_SR)

            chunk = normalize_audio(chunk.astype(np.float32))
            filename = f"{patient_name}_epoch{i+1:04d}.wav"
            sf.write(os.path.join(patient_dir, filename), chunk, TARGET_SR)
            results.append((filename, label))

        del audio_16k
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


def split_data(all_results):
    """
    환자 단위 6:2:2 분할.
    test는 항상 클래스 균형 맞춤.
    반환: (train_data, val_data, test_data)
    """
    by_patient = defaultdict(list)
    for filename, label in all_results:
        pnum = int(filename.split("_")[0].replace("patient", ""))
        by_patient[pnum].append((filename, label))

    nums = sorted(by_patient.keys())
    n = len(nums)
    n_train = max(1, int(n * 0.6))
    n_val = max(1, int(n * 0.2))

    train_p = nums[:n_train]
    val_p = nums[n_train:n_train + n_val]
    test_p = nums[n_train + n_val:]

    if not test_p and val_p:
        test_p = [val_p.pop()]
    if not val_p and len(train_p) > 2:
        val_p = [train_p.pop()]

    def collect(pnums):
        out = []
        for p in pnums:
            out.extend(by_patient[p])
        return out

    train_d, val_d = collect(train_p), collect(val_p)
    test_d_full = collect(test_p)

    # test는 항상 클래스 균형 맞춤
    test_d = balance_data(test_d_full)

    print(f"\n  분할 (환자 단위):")
    print(f"    train : patient {train_p}")
    print(f"    val   : patient {val_p}")
    print(f"    test  : patient {test_p}")
    print(f"    train.csv: {len(train_d)}개 {label_count_str(train_d)}")
    print(f"    val.csv: {len(val_d)}개 {label_count_str(val_d)}")
    print(f"    test.csv (원본): {len(test_d_full)}개 {label_count_str(test_d_full)}")
    print(f"    test.csv (균형): {len(test_d)}개 {label_count_str(test_d)}")

    return train_d, val_d, test_d


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
        if split_name == "test":
            data_2class = balance_data(data_2class)

        write_csv(os.path.join(dst_dir, f"{split_name}.csv"), data_2class)

        counts = {LABEL_NAMES_2CLASS.get(k, k): v for k, v in sorted(Counter(r[1] for r in data_2class).items())}
        print(f"  {split_name}.csv: {len(data_2class)}개 {counts}")


# ──────────────────────────────────────────────
# 클래스 밸런싱 (3-Class)
# ──────────────────────────────────────────────
def create_balanced_version(train_data, val_data, test_data, full_dir, ratio_dir):
    """
    각 split에서 rem/nrem/wake 비율을 최소 클래스에 맞춰 동일하게 만든다.
    선택된 WAV를 ratio_ver/ 에 환자별 폴더로 복사.
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

        # WAV 복사 (환자별 폴더 유지)
        for filename, label in balanced:
            patient_folder = filename.split("_")[0]
            src = os.path.join(full_dir, patient_folder, filename)
            dst_dir = os.path.join(ratio_dir, patient_folder)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        write_csv(os.path.join(ratio_dir, f"{split_name}.csv"), balanced)


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

    # 원본 백업 (신규만)
    print(f"\n{'='*60}")
    print("원본 파일 정리 → data_for_saving/")
    for i, (pid, rml, edfs) in enumerate(new_patients):
        num = next_num + i
        dest = os.path.join(SAVING_DIR, f"patient{num:02d}_{pid}")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(rml, dest)
        for e in edfs:
            shutil.copy2(e, dest)
        print(f"  patient{num:02d}: 1 RML + {len(edfs)} EDF → {os.path.basename(dest)}/")

    # full_ver: 신규 환자만 처리
    os.makedirs(FULL_DIR, exist_ok=True)
    new_results = []
    for i, (pid, rml, edfs) in enumerate(new_patients):
        num = next_num + i
        results = process_patient(pid, rml, edfs, num, FULL_DIR)
        new_results.extend(results)
        # 매니페스트에 추가
        processed[pid] = num

    # 매니페스트 저장 (처리 직후 즉시 저장)
    manifest["last_num"] = next_num + len(new_patients) - 1
    manifest["patients"] = processed
    save_manifest(manifest)

    print(f"\n{'='*60}")
    named = {LABEL_NAMES.get(k, k): v for k, v in sorted(Counter(r[1] for r in new_results).items())}
    print(f"신규 처리: {len(new_results)}개 {named}")

    if not new_results:
        print("처리된 데이터 없음!")
        return

    # CSV 생성 (현재 로컬에 있는 데이터만)
    print(f"\n{'='*60}")
    print("full_ver CSV 생성 (신규 데이터)")
    train_d, val_d, test_d = split_data(new_results)
    write_csv(os.path.join(FULL_DIR, "train.csv"), train_d)
    write_csv(os.path.join(FULL_DIR, "val.csv"), val_d)
    write_csv(os.path.join(FULL_DIR, "test.csv"), test_d)

    # ratio_ver (3-Class 균형)
    if os.path.exists(RATIO_DIR):
        shutil.rmtree(RATIO_DIR)
    os.makedirs(RATIO_DIR, exist_ok=True)
    create_balanced_version(train_d, val_d, test_d, FULL_DIR, RATIO_DIR)

    # full_ver_2class (2-Class, test 균형)
    if os.path.exists(FULL_2CLASS_DIR):
        shutil.rmtree(FULL_2CLASS_DIR)
    create_2class_version(train_d, val_d, test_d, FULL_DIR, FULL_2CLASS_DIR)

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
    print(f"  data_for_saving:          {SAVING_DIR}")
    print(f"  매니페스트:               {MANIFEST_PATH}")
    print(f"  총 누적 환자:             {manifest['last_num']}명")
    print("=" * 60)


if __name__ == "__main__":
    main()
