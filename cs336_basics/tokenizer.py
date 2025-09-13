# In cs336_basics/tokenizer.py

import multiprocessing
from collections import Counter
import regex as re

def train_bpe(input_path: str, vocab_size: int, special_tokens: list [str]):
    """
    Trains a BPE tokenizer from a text file.

    Args:
        input_path: The path to the text file to train on.
        vocab_size: The desired final vocabulary size.
        special_tokens: A list of special tokens to add to the vocabulary.

    Returns:
        A tuple containing:
        - vocab: A dictionary mapping token IDs (int) to token bytes (bytes).
        - merges: A list of merged byte pairs.
    """

    #--- Step 1: Initialization ---
    
    vocab = {}
    merges = []

    # Initialize the vocabulary with base byte tokens
    vocab = {i : bytes([i]) for i in range (256)}

    # Add speical tokens to your vocabulary
    for i, token_str in enumerate(special_tokens):
        vocab[256+i] = token_str.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    # read raw training text from the file

    print(f"Reading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    #--- Step 2: Remove special tokens and pre-tokrnization ---

    # ‘escape’ the special tokens so special characters are treated as literal characters 
    special_pattern = "|".join(re.escape(s) for s in special_tokens)
    # divide the `text` string into a list of substrings
    text_chunks = re.split(f"({special_pattern})",text)

    text_only_chunks = [chunk for chunk in text_chunks if chunk and chunk not in special_tokens]

    # Create a pool of processes and map the worker function to the chunks
    with multiprocessing.Pool() as pool:
        list_of_freq_dicts = pool.map(process_text_chunk, text_only_chunks)

    word_freqs = Counter()
    for freq_dict in list_of_freq_dicts:
        word_freqs.update(freq_dict)

    word_freqs = dict(word_freqs)


    # Represent words as tuples of their bytes for easier processing
    # e.g., "hello" -> (b'h', b'e', b'l', b'l', b'o')
    # This makes them hashable so they can be dictionary keys.
    byte_word_freqs = {}
    for word_str ,freq in word_freqs.items():
        # We add a special end-of-word symbol to handle merges at the end of words
        # This is a common practice, though not explicitly required by the assignment text.
        # For now, let's keep it simple as the assignment implies.
        word_tuple = tuple(bytes([b]) for b in word_str.encode('utf-8'))
        byte_word_freqs[word_tuple] = freq

    word_freqs = byte_word_freqs
    
    # --- Step 3: Initialize Pair Counts ---
    pair_counts = {}
    stats = {}

    for word_tuple, freq in word_freqs.items():
        # --- Logic for building pair_counts ---
        for i in range (len(word_tuple)-1):
            pair = (word_tuple[i], word_tuple[i+1])
            # Add the words' frequency to the pair's count
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
        # --- Logic for building stats (inverted index) ---
        for token in set(word_tuple):
            if token not in stats:
                stats[token] = set()
            stats[token].add(word_tuple)

    
    # --- Step 4: The Main Merge Loop ---
    for i in range(num_merges):
        if not pair_counts:
            break

        # Find the highest freqency values 
        max_freq = max(pair_counts.values())
        # Get all pairs that are tied for this frequency
        tied_pairs = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        # From the tied pairs, choose the one that is lexicographically largest.
        best_pair = max(tied_pairs)

        # update merges and vocab
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1] 
        vocab[len(vocab)] = new_token
        stats[new_token] = set()
        
        # --- Use the index to get only the affected words ---
        token_A, token_B = best_pair
        # Look for the words have token_A or token_B
        affected_words = stats.get(token_A, set()) | stats.get(token_B, set())

        for word_tuple in affected_words: 

            freq = word_freqs[word_tuple]

            del word_freqs[word_tuple]

            # Merge the selected best pair in words
            new_word_tuple = merge_pair_in_word(word_tuple, best_pair, new_token)

            # Update word_freqs in place for next iteration 
            word_freqs[new_word_tuple] = freq

            update_pair_counts(pair_counts, word_tuple, new_word_tuple, freq)
            update_stats(stats, word_tuple, new_word_tuple)
                    
    return vocab, merges

def merge_pair_in_word(word_tuple, pair_to_merge, new_token):

    new_word_parts = []

    i=0
    while i<len(word_tuple):
        # Look for the pair at the current position 'i'
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == pair_to_merge:
            # If we find the pair, add the new merged token
            new_word_parts.append(new_token)
            # And skip our index ahead by 2
            i += 2

        else:
            # If we don't find the pair, just add the current token
            new_word_parts.append(word_tuple[i])
            # And move our index ahead by 1
            i += 1

    # Since lists can be changed, they cannot be used as dictionary keys. 
    # Tuples, being unchangeable, are perfect for this job.
    return tuple(new_word_parts)

def update_pair_counts(pair_counts, old_word, new_word, freq):
    """
    Updates pair counts and the priority queue after a word is modified.
    """
    # Step 1: Decrement/remove the counts for all pairs from the OLD word.
    # This effectively removes the old word's contribution.
    for i in range(len(old_word)-1):
        pair = (old_word[i], old_word[i+1])
        if pair in pair_counts:
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
    
    # Step 2: Increment/add the counts for all pairs from the NEW word.
    for i in range(len(new_word)-1):
        pair = (new_word[i], new_word[i+1])
        # Update the count in main dictionary
        pair_counts[pair] = pair_counts.get(pair, 0) + freq

    pass

def update_stats(stats, old_word_tuple, new_word_tuple):
    """
    Updates the inverted index (stats) after a word is modified.
    """
    # Step 1: Remove the old word from the index for each of its tokens.
    for token in set(old_word_tuple):
        # We use .discard() because it's safer than .remove().
        # It won't cause an error if the item isn't in the set.
        if token in stats:
            stats[token].discard(old_word_tuple)

    # Step 2: Add the new word to the index for each of its tokens.
    for token in set(new_word_tuple):
        if token not in stats:
            stats[token] = set()
        stats[token].add(new_word_tuple)

# This is the "worker" function that will run in parallel
def process_text_chunk(text_chunk):
    """Take a chunk of text and returns a dictionary of word frequencies"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freqs = Counter()

    words = re.findall(PAT, text_chunk)
    for word in words:
        word_freqs[word]+=1

    return word_freqs


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        # We'll build the setup logic here.
        pass

    # We will add other methods like encode() and decode() later.