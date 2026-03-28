sample_rate = 16000
clip_samples = sample_rate * 30  # 30초 = 480,000 샘플

# 2클래스: Sleep (rem+nrem), Wake
labels = ['sleep', 'wake']
classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}
ix_to_lb = {i: label for i, label in enumerate(labels)}
