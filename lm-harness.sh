. "/xpfs/public/infra/train/miniconda3/etc/profile.d/conda.sh"
conda activate /xpfs/public/infra/train/miniconda3/envs/jesper_sft
cd /xpfs/public/infra/train/jesper/LLaMA-Factory
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/xpfs/public/infra/train/jesper/LLaMA-Factory/saves/llama2-7b \
    --tasks triviaqa\
    --batch_size 4
    # --dst_mode joint\
    # --dst_ratio 1.0 \

#mmlu,boolq,piqa,hellaswag,winogrande,social_iqa,arc_challenge,arc_easy