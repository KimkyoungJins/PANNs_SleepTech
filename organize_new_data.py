#!/usr/bin/env python3
"""
new_data 폴더의 flat 구조(파일이 섞여 있음)를
환자별 서브디렉토리로 정리하는 스크립트.

변환:
  new_data/
    ├── 00000995-100507.rml
    ├── 00000995-100507[001].edf
    ├── 00000995-100507[002].edf
    ├── 00001014-100507.rml
    ├── 00001014-100507[001].edf
    └── ...

  → new_data/
    ├── 00000995-100507/
    │   ├── 00000995-100507.rml
    │   ├── 00000995-100507[001].edf
    │   └── 00000995-100507[002].edf
    ├── 00001014-100507/
    │   ├── 00001014-100507.rml
    │   └── 00001014-100507[001].edf
    └── ...

특징:
  - 기본은 "이동"(mv) 모드 (용량 절약)
  - --copy 옵션: 복사 모드 (원본 유지)
  - --dry-run 옵션: 실제 변경 없이 계획만 출력
  - RML이 없는 환자도 폴더 생성 (EDF만 있어도)
  - EDF가 없는 환자는 건너뜀
  - 이미 서브디렉토리에 있는 파일은 건드리지 않음
  - labels_xxx.csv 같은 기타 파일은 무시
"""

import os
import shutil
import argparse
import re
from collections import defaultdict


NEW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_data")


def extract_patient_id(filename):
    """파일명에서 환자 ID 추출.

    예:
      "00000995-100507.rml"       → "00000995-100507"
      "00000995-100507[001].edf"  → "00000995-100507"
      "labels_00001008-100507.csv" → None (기타 파일)
    """
    # RML 파일
    if filename.endswith(".rml"):
        pid = filename[:-4]  # .rml 제거
        if re.match(r"^\d{8}-\d{6}$", pid):
            return pid
        return None

    # EDF 파일 (형식: {pid}[NNN].edf)
    if filename.endswith(".edf"):
        match = re.match(r"^(\d{8}-\d{6})\[\d+\]\.edf$", filename)
        if match:
            return match.group(1)
        return None

    return None


def group_files_by_patient(data_dir):
    """flat 구조의 파일들을 환자 ID별로 그룹화.

    Returns:
        dict: {pid: {"rml": rml_file_or_None, "edf": [edf_files]}}
    """
    groups = defaultdict(lambda: {"rml": None, "edf": []})
    ignored = []

    for filename in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, filename)

        # 디렉토리는 건너뜀 (이미 정리됨)
        if os.path.isdir(full_path):
            continue

        pid = extract_patient_id(filename)
        if pid is None:
            ignored.append(filename)
            continue

        if filename.endswith(".rml"):
            groups[pid]["rml"] = filename
        elif filename.endswith(".edf"):
            groups[pid]["edf"].append(filename)

    # EDF 파일 정렬
    for pid in groups:
        groups[pid]["edf"].sort()

    return groups, ignored


def organize_patient(pid, files, data_dir, mode="move", dry_run=False):
    """한 환자의 파일들을 해당 환자 폴더로 이동/복사.

    Args:
        pid: 환자 ID (폴더 이름이 됨)
        files: {"rml": filename, "edf": [filenames]}
        data_dir: new_data 디렉토리
        mode: "move" 또는 "copy"
        dry_run: True면 실제 변경 없이 로그만
    """
    patient_dir = os.path.join(data_dir, pid)

    rml = files["rml"]
    edfs = files["edf"]

    actions = []

    # 대상 폴더 생성
    if not dry_run:
        os.makedirs(patient_dir, exist_ok=True)

    # RML 이동/복사
    if rml:
        src = os.path.join(data_dir, rml)
        dst = os.path.join(patient_dir, rml)
        if not os.path.exists(dst):
            if not dry_run:
                if mode == "move":
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
            actions.append(f"{mode.upper()} {rml}")

    # EDF 이동/복사
    for edf in edfs:
        src = os.path.join(data_dir, edf)
        dst = os.path.join(patient_dir, edf)
        if not os.path.exists(dst):
            if not dry_run:
                if mode == "move":
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
            actions.append(f"{mode.upper()} {edf}")

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="new_data 폴더를 환자별 서브디렉토리로 정리"
    )
    parser.add_argument("--data_dir", default=NEW_DATA_DIR,
                        help="정리할 폴더 (기본: new_data)")
    parser.add_argument("--copy", action="store_true",
                        help="이동 대신 복사 (원본 유지)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 계획만 출력")
    args = parser.parse_args()

    data_dir = args.data_dir
    mode = "copy" if args.copy else "move"

    if not os.path.isdir(data_dir):
        print(f"[ERROR] 디렉토리 없음: {data_dir}")
        return 1

    print("=" * 60)
    print(f"new_data 정리 시작")
    print(f"  경로: {data_dir}")
    print(f"  모드: {mode}{' (DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    # 파일 그룹화
    groups, ignored = group_files_by_patient(data_dir)

    if not groups:
        print("\n처리할 파일이 없습니다.")
        return 0

    # 환자별 요약 (정렬)
    sorted_pids = sorted(groups.keys())

    print(f"\n환자 {len(sorted_pids)}명 발견")
    print("-" * 60)

    complete = []        # RML + EDF 모두 있음
    edf_only = []        # EDF만 있음 (RML 없음)
    rml_only = []        # RML만 있음 (EDF 없음)

    for pid in sorted_pids:
        files = groups[pid]
        rml_ok = files["rml"] is not None
        edf_count = len(files["edf"])

        if rml_ok and edf_count > 0:
            complete.append(pid)
            status = f"✅ RML + EDF {edf_count}개"
        elif edf_count > 0:
            edf_only.append(pid)
            status = f"⚠️  EDF {edf_count}개만 (RML 없음)"
        else:
            rml_only.append(pid)
            status = f"❌ RML만 (EDF 없음)"

        print(f"  {pid}: {status}")

    print("-" * 60)
    print(f"  완전한 환자 (RML+EDF): {len(complete)}명")
    print(f"  EDF만 있음:            {len(edf_only)}명")
    print(f"  RML만 있음 (제외):     {len(rml_only)}명")

    if ignored:
        print(f"\n무시된 파일 ({len(ignored)}개):")
        for f in ignored[:10]:
            print(f"    {f}")
        if len(ignored) > 10:
            print(f"    ... 외 {len(ignored) - 10}개")

    # 이동/복사 실행
    target_pids = complete + edf_only  # RML만 있는 건 제외

    if not target_pids:
        print("\n정리할 환자가 없습니다.")
        return 0

    print(f"\n{'=' * 60}")
    print(f"{len(target_pids)}명 환자 파일 {mode}{' (DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    total_actions = 0
    for pid in target_pids:
        actions = organize_patient(pid, groups[pid], data_dir,
                                    mode=mode, dry_run=args.dry_run)
        if actions:
            print(f"\n{pid}/ ({len(actions)}개)")
            for a in actions[:3]:
                print(f"  {a}")
            if len(actions) > 3:
                print(f"  ... 외 {len(actions) - 3}개")
            total_actions += len(actions)

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"DRY RUN 완료 - 실제 변경 없음")
        print(f"  이동/복사 예정: {total_actions}개 파일")
        print(f"\n실제 실행하려면 --dry-run 없이 다시 실행하세요.")
    else:
        print(f"완료!")
        print(f"  처리된 파일: {total_actions}개")
        print(f"  생성된 폴더: {len(target_pids)}개")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
