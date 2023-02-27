python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
    ./tools/train.py configs/rotated_dab_detr/rotated_dab_detr_r50_4x2_dota_oc.py \
    --launcher pytorch \
    --work-dir /media/titan/G/Guru/output//rotated_dab_detr_r50_4x2_dota_oc_new \