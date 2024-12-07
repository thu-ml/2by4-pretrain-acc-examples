export CUDA_VISIBLE_DEVICES=0

hf_model=gpt2

python run_qa_no_trainer.py \
   --model_name_or_path $hf_model \
   --dataset_name squad \
   --max_seq_length 384 \
   --doc_stride 128 \
   --output_dir "../out/squad/" \
   --with_tracking \
   --learning_rate 5e-6 \
   --num_warmup_steps 0.1 \
   --lr_scheduler_type cosine