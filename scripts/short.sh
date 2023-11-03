#! /bin/bash

position_types=( "sinusoidal" "rotary" "alibi" "relative" ) # "dynamic" )

for position_type in "${position_types[@]}"; do
    python experiment_runner.py \
        --length short \
        --language java \
        --position_type $position_type \
        --output_dir /semeru/completeformer/models \
        --batch_size 128
done