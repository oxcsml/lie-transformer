source venv/bin/activate

tasks=(homo lumo gap alpha mu Cv G H r2 U U0 zpve)

if [ $# == 0 ]
then
    task=${tasks[0]}
elif [ $# == 1 ]
then
    task=${tasks[$1]}
elif [ $# == 2 ]
then
    task=${tasks[$1]}
    export CUDA_VISIBLE_DEVICES=$2
fi

echo Training task $task on GPU $CUDA_VISIBLE_DEVICES

python scripts/train_molecule.py \
    --run_name "all_tasks" \
    --model_config "configs/molecule/lie_resnet.py" \
    --learning_rate 3e-3 \
    --train_epochs 500 \
    --batch_size 75 \
    --data_augmentation True \
    --fill 0.5 \
    --lr_schedule cosine_warmup \
    --channels 1536 \
    --task $task \
    --resume True
    # --parameter_count True
