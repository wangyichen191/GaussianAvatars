CUDA_VISIBLE_DEVICES=3 ../anaconda3/envs/gaussian-avatars/bin/python3 train.py \
-s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_218_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/AllOffset_originMLP86_changedScalingLoss \
--port 60010 --eval --white_background --bind_to_mesh --not_finetune_flame_params