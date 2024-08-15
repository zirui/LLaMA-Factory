accelerate launch -m lm_eval --model hf \
    --model_args pretrained=saves/llama2-7b \
    --tasks boolq,piqa,hellaswag,winogrande\
    --dst_mode oracle\
    --dst_ratio 1.0 \
    --batch_size 4