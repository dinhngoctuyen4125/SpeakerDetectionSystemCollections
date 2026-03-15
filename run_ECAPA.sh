read DATA_ROOT NOISE_DIR < <(python get_dataset.py \
  --datasets "datasets")

echo "DATA_ROOT: $DATA_ROOT"
echo "NOISE_DIR: $NOISE_DIR"

# python train.py \
#   --data_root "$DATA_ROOT" \
#   --noise_dir "$NOISE_DIR"