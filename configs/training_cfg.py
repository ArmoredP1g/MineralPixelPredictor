device = "cuda"
dataset_path = "D:\\IR_DATA"
ckpt_path = "ckpt"
step_per_fold = 100000
batch_size = 96
learning_rate = 1e-4
lr_decay=0.93
lr_decay_step=1000
lr_lower_bound=1e-7
num_workers=6
# encoder_type = "MSI"
# norm_type = "min_max"
session_tag = "Conv_Diff"