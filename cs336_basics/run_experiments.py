import numpy as np
import time
import os
from cs336_basics.tokenizer import Tokenizer

def sample_documents(filepath: str, num_samples: int =10, delimiter: str = "<|endoftext|>"):
    """Reads a text file and randomly samples `num_samples` documents."""
    with open(filepath, 'r', encoding = 'utf-8') as f:
        text = f.read()
    documents = text.split(delimiter)
    sampled_indices = np.random.choice(len(documents), min(num_samples, len(documents)), replace = False)
    return [documents[i] for i in sampled_indices]

def calculate_compression(tokenizer: Tokenizer, text_sample: list[str]):
    """Calculate the compresssion ratio (bytes/token) for a tokenizer on a sample of text """
    full_text = "".join(text_sample)
    
    total_bytes = len(full_text.encode('utf-8'))
    total_tokens = len(tokenizer.encode(full_text))
    
    if total_tokens == 0:
        return 0
    
    return total_bytes / total_tokens


def measure_throughput (filepath: str, vocab_filepath: str = "" ,
                merges_filepath: str = "", special_tokens: list[str] = ["<|endoftext|>"],
                output_filepath: str = None):
    
    """Loads a tokenizer, encodes a file, and prints the throughput."""
    
    with open(filepath, 'r', encoding = 'utf-8') as f:
        text = f.read()
        
    num_bytes = len(text.encode('utf-8'))
    
    # 2. Load the tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath = vocab_filepath,
        merges_filepath=merges_filepath, 
        special_tokens=special_tokens
    )
    
    # Time the encode operation
    
    start_time = time.time()
    token_ids = tokenizer.encode(text=text)
    end_time = time.time()
    
    duration = end_time - start_time
    throughput = num_bytes / duration 
    
    print(f"Encoded {num_bytes / 1e6:.2f} MB in {duration:.3f} seconds. ")
    print(f"Throughput: {throughput / 1e6:.2f} MB/s")
    
    if (output_filepath is not None):
        token_array = np.array(token_ids, dtype = np.uint16)
        print(f"Saved {len(token_array)} tokens to {output_filepath}...")
        np.save(output_filepath, token_array)
    
    return throughput

def main():
    # --- Part(a) ---
    # 1. Load your trained TinyStories tokenizer
    print("Loading TinyStories tokenizer (10k vocab) ...")
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath="results/tinystories_vocab.json",
        merges_filepath="results/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    
    # 2. Sample data from TinyStories
    print("Sampling from TinyStories dataset...")
    ts_sample = sample_documents("data/TinyStoriesV2-GPT4-train.txt")
    
    # 3. Calculate and print the result
    ts_compression = calculate_compression(ts_tokenizer, ts_sample)
    print(f"TinyStories tokenizer on TinyStories data: {ts_compression:.2f} bytes/token")
    
    # --- You will add the logic for OpenWebText here ---
    print("Loading OpenWebText tokenizer (32k vocab)")
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="results/OpenWebText_vocab.json",
        merges_filepath="results/OpenWebText_merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    
    print("Sampling from OpenWebText dataset...")
    owt_sample = sample_documents("data/owt_train.txt")
    
    owt_compression = calculate_compression(owt_tokenizer, owt_sample)
    print(f"OpenWebText tokenizer on OpenWebText data: {owt_compression:.2f} bytes/token")
    
    
    # --- Part(b) ---
    print("Testing TinyStories tokenizer on OpenWebText data...")
    
    # Use the TinyStories tokenizer onthe OpenWebText sample
    cross_compression = calculate_compression(ts_tokenizer, owt_sample)
    print(f"TinyStories tokenizer on OpenWebText data: {cross_compression:.2f} bytes/token")
    
    print("Testing OpenWebText tokenizer on TinyStories data...")
    
    cross_compression = calculate_compression(owt_tokenizer, ts_sample)
    print(f"OpenWebText tokenizer on TinyStories data: {cross_compression:.2f} bytes/token")
    
    # --- Part(c) ---
    print("Testing the entire time used for encodingthe TinyStories and OpenWebText../")
    
    measure_throughput(filepath="data/TinyStoriesV2-GPT4-train.txt", vocab_filepath="results/tinystories_vocab.json",
               merges_filepath="results/tinystories_merges.pkl", output_filepath= "data/tinystories_train.npy")
    
    measure_throughput(filepath="data/owt_train.txt", vocab_filepath= "results/OpenWebText_vocab.json",
               merges_filepath="results/OpenWebText_merges.pkl", output_filepath= "data/OpenWebText_train.npy")
    
    
if __name__ == "__main__":
    main()       