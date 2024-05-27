#
python eval.py eval=GLUE_sane name="transposable SR-STE decay\=0. in_Adam" eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True wandb=default wandb.project=performance-eval

python eval.py eval=GLUE_sane name="transposable SR-STE+MVUE24 decay\=6e-6 in_Adam" eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True wandb=default wandb.project=performance-eval

python eval.py eval=GLUE_sane name="transposable SR-STE+MVUE24+dense-finetune decay\=6e-6 in_Adam" eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True wandb=default wandb.project=performance-eval

