#!/bin/bash
. "/xpfs/public/infra/train/miniconda3/etc/profile.d/conda.sh"
conda activate /xpfs/public/infra/train/miniconda3/envs/jesper_sft
cd /xpfs/public/infra/train/jesper/LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml