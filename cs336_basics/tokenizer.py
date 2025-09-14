# In cs336_basics/tokenizer.py

import multiprocessing
from collections import Counter
import regex as re
import pickle
import json
from typing import Iterable, Iterator

# --- MAIN TRAINING FUNCTION ---

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

# --- HELPER FUNCTIONS FOR TRAINING ---

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

# Find the best pair with the least merge rank 
def find_best_pair(tokens, mergeranks):
    least_pair_rank = float('inf')
    best_pair = None
    
    for i in range(len(tokens)-1):
        current_pair = (tokens[i], tokens[i+1])
        if (current_pair in mergeranks) and (mergeranks[current_pair] < least_pair_rank):
            least_pair_rank = mergeranks[current_pair]
            best_pair = current_pair
            
    return best_pair
    
# Merge the best pair into a new list of tokens
def merge_the_pair(tokens, best_pair):
    new_token_list = []
    new_token = best_pair[0] + best_pair[1]
    i = 0
    while i < len(tokens):
        # Check if the current and next tokens is the best_pair
        if (i < len(tokens) -1 ) and ((tokens[i], tokens[i+1]) ==best_pair):
            new_token_list.append(new_token)
            i+=2
        else:
            new_token_list.append(tokens[i])
            i+=1
        
    return new_token_list
            

# --- THE TOKENIZER CLASS ---

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        
        # 1. Store the vocabulary for DECODING (ID -> bytes)
        self.vocab = vocab
        # Store PAT
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 2. Create and store the inverted vocabulary for ENCODING (bytes -> ID)
        self.encoder = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        # 3. Create and store a dictionary for fast merge lookups (pair -> rank)
        # The rank is just the position in the ordered merge list 
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # 4. Handle special tokens
        self.special_tokens = {} # str -> bytes
        if special_tokens:
            # sort the tokens from longest to shortest
            special_tokens.sort(key=len, reverse=True)
            
            for token_str in special_tokens:
                self.special_tokens[token_str] = token_str.encode('utf-8')

        # 5. Create a pre-compiled  regex to find special tokens during encoding
        # This is much faster than searching for them manually
        if self.special_tokens:
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            self.special_tokens_pattern = re.compile(special_pattern)
        else:
            self.special_tokens_pattern = None
    
    def encode(self, text: str)-> list[int]:
        final_token_ids = []

        if self.special_tokens_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_tokens_pattern.split(text)
        
        for chunk in chunks:
            if chunk in self.special_tokens:
                final_token_ids.append(self.encoder[self.special_tokens[chunk]])
            else:
                # 1. Pre-tokenize the chunk into smaller words
                words = re.findall(self.pat, chunk)

                for word in words:
                    tokens = [bytes([b]) for b in word.encode('utf-8')]

                    # Loop until no more merges can be applied
                    while len(tokens) >1:
                        # find the best pair to merge in the current `tokens` list
                        best_pair = find_best_pair(tokens, self.merge_ranks)

                        if best_pair is None:
                            break # No more merges are possible in the word

                        tokens = merge_the_pair(tokens, best_pair)

                    for token in tokens: 
                        final_token_ids.append(self.encoder[token])


        return final_token_ids

    def decode(self, ids: list[int]) -> str:
        
        # Get all the bytes pieces from the vocabulary
        byte_pieces = [self.vocab[token_id] for token_id in ids]
        
        full_byte_sequence = b"".join(byte_pieces)
        
        return full_byte_sequence.decode('utf-8', errors='replace')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings, yielding token IDs one by one.
        """
        # Loop through each piece of text from the iterable (e.g., each line from a file)
        for text_chunk in iterable:
            # Use your existing encode method on the chunk and yield each ID from the result
            yield from self.encode(text_chunk)
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Constructs a Tokenizer from saved vocab and merges files.
        """
        # 1. Load the vocabulary from the JSON files. 
        print(f"Loading vocabulary from {vocab_filepath} ...")
        with open(vocab_filepath, 'r') as f:
            serializable_vocab = json.load(f)
            # Convert the saved lists of integers back into bytes objects
            vocab = {int(k): bytes(v) for k, v in serializable_vocab.items()}
            
        # 2. Load the merges from the pickle file
        print(f"Loading merges from {merges_filepath}")
        with open (merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        # 3. Call the class's own __init__ method to create a new instance
        return cls(vocab= vocab, merges = merges, special_tokens = special_tokens)
    

    