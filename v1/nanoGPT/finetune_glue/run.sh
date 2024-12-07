export CUDA_VISIBLE_DEVICES=0

hf_model=gpt2

export TASK_NAME=cola

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=sst2

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=stsb

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=qqp

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=mnli

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=qnli

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine


export TASK_NAME=rte

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine

export TASK_NAME=wnli

python run_glue_no_trainer.py \
  --model_name_or_path $hf_model \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --output_dir "../out/$TASK_NAME/" \
  --num_warmup_steps 0.1 \
  --lr_scheduler_type cosine