uv run python cs336_basics/run_bpe_training.py \
    --input_path data/owt_train.txt \
    --vocab_size 32000 \
    --output_vocab_file results/OpenWebText_vocab.json \
    --output_merges_file results/OpenWebText_merges.pkl