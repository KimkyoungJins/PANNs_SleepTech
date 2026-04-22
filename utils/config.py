sample_rate = 16000
clip_samples = sample_rate * 30  # 30초 = 480,000 샘플

# 2클래스: Wake=0, Sleep=1 (rem+nrem → sleep)
labels = ['wake', 'sleep']
classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}  # {'wake':0, 'sleep':1}
ix_to_lb = {i: label for i, label in enumerate(labels)}  # {0:'wake', 1:'sleep'}
