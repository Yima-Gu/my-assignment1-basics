import json
import pickle
import time 
import argparse
from cs336_basics.tokenizer import train_bpe 


def main():

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a text file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--vocab_size", type=int, required=True, help="The desired final vocabulary size.")
    parser.add_argument("--output_vocab_file", type=str, required=True, help="Path to save the final vocabulary.")
    parser.add_argument("--output_merges_file", type=str, required=True, help="Path to save the final merges.")
    parser.add_argument("--special_tokens", type=str, nargs='+', default=["<|endoftext|>"], help="A list of special tokens.")

    args = parser.parse_args()

    # --- Settings for Problem (train_bpe_tinystories) ---
    print("--- Training tokenizer on TinyStories ---")

    # Run the Training
    start_time = time.time()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)
    end_time = time.time()

    print(f"Training finished in {end_time - start_time :.2f} seconds. ")

    serializable_vocab = {token_id: list(token_bytes) for token_id, token_bytes in vocab.items()}

    print(f"Saving vocabulary to {args.output_vocab_file}")
    with open(args.output_vocab_file, 'w' ) as f:
        # We save the vocab in a human-readable way
        json.dump(serializable_vocab , f )

    print(f"Saving merges to {args.output_merges_file}")
    with open(args.output_merges_file, 'wb') as f:
        pickle.dump(merges, f)
    
    print("Done!")

if __name__ == "__main__":
    main()