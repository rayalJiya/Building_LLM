# Example text
texts = [
    "main ghar ja raha hoon",
    "bpe tokenizer bahut useful hai",
    "python se data pairs banana easy hai"
]

def create_input_target_pairs(text, tokenizer=None):
    """
    Simple character-level example.
    Real mein aap BPE tokenizer use kar sakte ho.
    """
    # Character level (demo ke liye)
    tokens = list(text)          # ['m', 'a', 'i', 'n', ' ', ...]
    
    # Agar aapke paas BPE tokenizer hai to:
    # tokens = tokenizer.encode(text)
    
    input_ids = tokens[:-1]      # last token hata do
    target_ids = tokens[1:]      # pehla token hata do (shift)
    
    return input_ids, target_ids

# Saare texts ke pairs banao
all_pairs = []
for text in texts:
    inp, tgt = create_input_target_pairs(text)
    all_pairs.append((inp, tgt))

# Dekho kaisa dikhta hai
for i, (inp, tgt) in enumerate(all_pairs):
    print(f"\nExample {i+1}")
    print("Input :", "".join(inp))
    print("Target:", "".join(tgt))