#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v cargo >/dev/null 2>&1; then
    if [ -x "$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo" ]; then
        export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
    fi
fi

WORK="${ROOT_DIR}/.tmp/e2e"
rm -rf "$WORK"
mkdir -p "$WORK"

cargo build --example gen_fixture --features backend-candle >/dev/null
./target/debug/examples/gen_fixture "$WORK"

mkdir -p "$WORK/model" 2>/dev/null || true

# prepare dataset
cargo run -- prepare-dataset \
  --input "$WORK/train.jsonl" \
  --output "$WORK/dataset" \
  --tokenizer "$WORK/tokenizer.json" >/dev/null

cat > "$WORK/run.toml" <<EOF
[model]
base_path = "$WORK/model"
backend = "candle"
tokenizer = "$WORK/tokenizer.json"

[lora]
enabled = true
use_qlora = false
r = 2
alpha = 4
dropout = 0.0
target_modules = ["lm_head"]

[training]
device = "cpu"
batch_size = 2
gradient_accumulation = 1
lr = 0.001
epochs = 1
max_steps = 50
warmup_steps = 5
weight_decay = 0.0
seed = 7
output_dir = "$WORK/out"

[data]
train_path = "$WORK/dataset/train"
val_path = "$WORK/dataset/val"
max_seq_len = 128
shuffle = true
truncation = false

[evaluation]
metrics = ["perplexity", "exact_match", "bleu", "rouge_l"]
EOF

cargo run --features backend-candle -- train --config "$WORK/run.toml" >/dev/null

cargo run --features backend-candle -- eval \
  --config "$WORK/run.toml" \
  --dataset "$WORK/train.jsonl" >/dev/null

# resume path
cargo run --features backend-candle -- train \
  --config "$WORK/run.toml" \
  --resume-from "$WORK/out/checkpoint" >/dev/null

cargo run --features backend-candle -- merge \
  --base "$WORK/model" \
  --adapter "$WORK/out/adapter.safetensors" \
  --output "$WORK/merged" >/dev/null

# loss must fall
python3 - "$WORK/out/train_metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if len(rows) < 2:
    raise SystemExit("not enough log rows")
first = rows[0]["loss"]
last = rows[-1]["loss"]
if not (last < first):
    print(f"loss did not fall: {first} -> {last}", file=sys.stderr)
    raise SystemExit(1)
print(f"loss fell: {first} -> {last}")
PY

echo "e2e demo passed"