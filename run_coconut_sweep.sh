#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_FILE="configs/gsm_coconut.yaml"
BACKUP_FILE="$(mktemp)"
cp "$CONFIG_FILE" "$BACKUP_FILE"

restore_config() {
  cp "$BACKUP_FILE" "$CONFIG_FILE"
  rm -f "$BACKUP_FILE"
}
trap restore_config EXIT

# Format per entry:
#   c_thought epochs_per_stage max_latent_stage num_epochs
SWEEP_CONFIGS=(
  "2 1 6 8"
  "2 2 6 16"
  "2 3 6 24"
)

for sweep_config in "${SWEEP_CONFIGS[@]}"; do
  read -r C_THOUGHT EPOCHS_PER_STAGE MAX_LATENT_STAGE NUM_EPOCHS <<< "$sweep_config"

  echo "============================================================"
  echo "Running c_thought=${C_THOUGHT}, epochs_per_stage=${EPOCHS_PER_STAGE}, max_latent_stage=${MAX_LATENT_STAGE}, num_epochs=${NUM_EPOCHS}"
  echo "============================================================"

  python - "$CONFIG_FILE" "$C_THOUGHT" "$EPOCHS_PER_STAGE" "$MAX_LATENT_STAGE" "$NUM_EPOCHS" <<'PY'
import sys
import yaml

config_path, c_thought, epochs_per_stage, max_latent_stage, num_epochs = sys.argv[1:]

with open(config_path) as f:
    config = yaml.safe_load(f)

config["c_thought"] = int(c_thought)
config["epochs_per_stage"] = int(epochs_per_stage)
config["max_latent_stage"] = int(max_latent_stage)
config["num_epochs"] = int(num_epochs)

with open(config_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

  python main.py configs/gsm_coconut.yaml
done
