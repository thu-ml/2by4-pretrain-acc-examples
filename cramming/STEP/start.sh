#
python pretrain.py name=STEP arch=bert-c5 train=bert-o3 train.batch_size=4096 \
 data=c4-subset-processed impl.microbatch_size=128 budget=24 train.scheduler=progress-triangle impl.no_jit_compilation=True \
 train.steps=300000 wandb=default
