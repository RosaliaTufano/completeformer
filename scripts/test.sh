# /bin/bash

python test_checkpoint.py \
    --checkpoint_path /semeru/completeformer/models/relative_short/checkpoints/final_checkpoint.ckpt \
    --length short \
    --language java \
    --position_type relative \
    --output_dir /semeru/completeformer/models \
    --batch_size 64