#! /bin/bash

position_types=( "sinusoidal" "rotary" "alibi" ) # "relative" )

for position_type in "${position_types[@]}"; do
    python experiment_runner.py \
        --length medium \
        --language java \
        --position_type $position_type \
        --output_dir /semeru/completeformer/models \
        --batch_size 64
done