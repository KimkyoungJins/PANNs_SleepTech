sample_rate = 16000
clip_samples = sample_rate * 30

# 3클래스: REM, NREM, Wake
labels = ['rem', 'nrem', 'wake']
classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}
ix_to_lb = {i: label for i, label in enumerate(labels)}
