CUDA_VISBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use-env --master_port=47773 main.py \
    --model deit_base_distilled_patch16_224 \
    --batch-size 128 \
    --epochs 1000 \
    --drw 900 \
    --weighted-distillation \
    --gpu 0 \
    --teacher-path "Enter teacher path" \
    --distillation-type hard \
    --data-path inat18 \
    --data-set INAT18 \
    --output_directory deit_out_inat \
    --student-transform 0 \
    --teacher-transform 0 \
    --teacher-model resnet50 \
    --teacher-size 224 \
    --paco --moco-t 0.2 --moco-k 8192 --moco-dim 128 \
    --experiment [deit_lt_paco_sam_inat] \
    --custom_model \
    --accum-iter 4 \
    --num_workers 16 \
    --dist-eval \
    # --log-results