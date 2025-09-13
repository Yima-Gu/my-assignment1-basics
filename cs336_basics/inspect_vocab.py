import json

VOCAB_FILE = "tinystories_vocab.josn"

# Load the saved vocabulary
with open(VOCAB_FILE, 'r') as f:
    serializable_vocab = json.load(f)
    vocab = {int(k): bytes(v) for k, v in serializable_vocab.items()}

# Find the longest token by length
longest_token = b''
for token_bytes in vocab.values():
    if len(token_bytes) > len(longest_token):
        longest_token = token_bytes

print(f"The longest token is: {longest_token}")
print(f"Its length is: {len(longest_token)} bytes")

# Try to decode it to see what it is as a string
try:
    print(f"Decoded: '{longest_token.decode('utf-8')}'")
except UnicodeDecodeError:
    print("The token cannot be decoded into a valid string on its own.")