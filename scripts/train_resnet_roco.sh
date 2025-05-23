python train_roco.py \
  --exp_name "resnet-diff-roco" \
  --epochs 30 \
  --dataset "roco" \
  --ve_name "resnet" \
  --use_gca \
  --constant_lr \
  --lr_ed 2e-4 \
  --lr_ve 1e-4 \
  --warmup_epochs 0 \
  --weight_decay 1e-6 \
  --channel_reduction 4 \
  --num_layers 3 \
  --save_path "results" \
  --batch_size 64 \
  --accum_steps 1 \
  --early_stopping 10 \
  --max_length 100 \
  --step_size 10 \
  --max_gen 100 \
  --hidden_size 512 \
  --fc_size 2048 \
  --contrastive_proj_dim 256 \
  --vocab_size 20000 \
  --delta1 1 \
  --delta2 0.3 \
  --topk 3 \
  --temperature 1 \
  --lambda_init 0.8 \
  --dropout 0.0 \
  --beam_width 3 \
  --encoder_size 2048 \
  --num_heads 8 \
  --diff_num_heads 4 \
  --num_workers 16 \
  --image_path "" \
  --ann_path "" \
  --project_root "/workspace/D-ExpertTransformer" \
  # --from_pretrained "results/resnet-diff-roco/iu_xray.pth" \



