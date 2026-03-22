import numpy as np

# ===== 수면 단계 분류 설정 =====

sample_rate = 16000                   # 샘플레이트: 16kHz (Cnn14_16k 모델용)
clip_samples = sample_rate * 30       # 오디오 클립 길이: 30초 = 480,000 샘플

# ===== 수면 단계 클래스 정의 =====
# 3개 클래스: REM, NREM, Wake
labels = ['rem', 'nrem', 'wake']
classes_num = len(labels)             # 총 클래스 수 = 3

# ===== 라벨 ↔ 인덱스 변환 딕셔너리 =====
# 사용 예: lb_to_ix['rem'] → 0,  ix_to_lb[0] → 'rem'
lb_to_ix = {label: i for i, label in enumerate(labels)}  # 라벨 이름 → 숫자 인덱스
ix_to_lb = {i: label for i, label in enumerate(labels)}  # 숫자 인덱스 → 라벨 이름
