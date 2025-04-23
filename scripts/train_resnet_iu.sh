python train_iu.py \
  --exp_name "resnet-iu" \
  --epochs 50 \
  --dataset "iu_xray" \
  --ve_name "resnet" \
  --use_diff \
  --use_gca \
  --use_learnable_tokens \
  --lr_ed 4e-4 \
  --lr_ve 1e-4 \
  --warmup_epochs 30 \
  --weight_decay 1e-5 \
  --channel_reduction 4 \
  --num_layers 3 \
  --save_path "results" \
  --batch_size 64 \
  --accum_steps 1 \
  --early_stopping 10 \
  --max_length 60 \
  --step_size 10 \
  --max_gen 100 \
  --hidden_size 512 \
  --fc_size 2048 \
  --contrastive_proj_dim 256 \
  --vocab_size 1642 \
  --delta1 1 \
  --delta2 0.3 \
  --topk 3 \
  --temperature 1 \
  --lambda_init 0.8 \
  --dropout 0.2 \
  --beam_width 3 \
  --encoder_size 2048 \
  --num_heads 8 \
  --diff_num_heads 4 \
  --num_workers 16 \
  --image_path "/data" \
  --ann_path "iu_xray" \
  --project_root "/workspace/D-ExpertTransformer" \
  # --from_pretrained "results/resnet-diff-iu-combine/iu_xray.pth" \
  # --eval \


