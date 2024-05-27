#
python pretrain.py name=half arch=bert-c5 train=bert-o3 train.batch_size=4096 \
 data=c4-subset-processed impl.microbatch_size=128 budget=24 train.scheduler=progress-triangle impl.no_jit_compilation=True \
 train.steps=300000 wandb=default arch.intermed_size=1536

#
python eval.py eval=GLUE_sane name=half eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True wandb=default wandb.project=cramming-eval-fast

