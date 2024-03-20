CUDA_VISIBLE_DEVICES=0 ../anaconda3/envs/gaussian-avatars/bin/python3 train.py \
-s /mnt/sda/wangyichen/server15_2/GaussianAvatars/data/302/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/302_woFinetune \
--port 60019 --eval --white_background --bind_to_mesh --not_finetune_flame_params --compute_offset_iteration -1 \
--position_offset_regularization 0.2 \
--rotation_offset_regularization 0.2 \
--scaling_offset_regularization 0.2