import os
import os
os.chdir("../")
opt_list = ["adam","sgd"]
lr_list = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3]
seed_list = [999, 189, 114, 929, 290, 848, 538, 874, 295, 266]
for opt in opt_list:
    for lr in lr_list:
        for seed in seed_list:
            os.system(f"python train_decoder.py -opt={opt} -lr={lr} -sd={seed} -ep=200")
