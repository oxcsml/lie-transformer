source venv/bin/activate

tasks=(homo lumo gap alpha mu Cv G H r2 U U0 zpve)

dataseed=0
modelseed=0
kernel=2232
layers=6
dim_hidden=1504
heads=8
lr=1e-3
kernel_dim=6
lr_schedule="cosine"
lr_floor=0
warmup_length=0.01
task=[ if $1 exists, ${tasks[$1]}, else, ${tasks[0]}]

python scripts/train_molecule.py \
    --run_name "limit_norm_test" \
    --model_config "configs/molecule/eqv_transformer_model.py" \
    --model_seed $modelseed \
    --data_seed $dataseed \
    --batch_fit 4000 \
    --task $task \
    --data_augmentation True \
    --learning_rate $lr \
    --lr_schedule $lr_schedule \
    --warmup_length $warmup_length \
    --lr_floor $lr_floor \
    --subsample_trainset 1.0 \
    --train_epochs 500 \
    --batch_size 100 \
    --num_heads $heads \
    --dim_hidden $dim_hidden \
    --num_layers $layers \
    --kernel_type $kernel \
    --kernel_dim $kernel_dim \
    --lift_samples 4 \
    --feature_embed_dim 8 \
    --mc_samples 25 \
    --fill 0.5 \
    --block_norm layer_pre \
    --kernel_norm none \
    --output_norm none \
    --architecture lieconv \
    --attention_fn dot_product \
    --parameter_count True \
    --max_sample_norm 1e6