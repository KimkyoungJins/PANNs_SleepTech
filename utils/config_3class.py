sample_rate = 16000
clip_samples = sample_rate * 30

# 3클래스: Wake=0, REM=1, NREM=2
labels = ['wake', 'rem', 'nrem']
classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}  # {'wake':0, 'rem':1, 'nrem':2}
ix_to_lb = {i: label for i, label in enumerate(labels)}  # {0:'wake', 1:'rem', 2:'nrem'}
