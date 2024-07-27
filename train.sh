CUDA_VISIBLE_DEVICES=4 python train.py \
-s /data2/wangyichen/SplattingAvatar/data/flame/yufeng \
-m output/yufeng_test \
--port 60019 --eval --white_background --bind_to_mesh --not_finetune_flame_params --compute_offset_iteration -1 \
--position_offset_regularization 0.2 \
--rotation_offset_regularization 0.2 \
--scaling_offset_regularization 0.2