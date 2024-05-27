#
python eval.py eval=GLUE_sane name="Bi-Mask" eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True wandb=default wandb.project=cramming-eval-fast

