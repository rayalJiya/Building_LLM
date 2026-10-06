class CharacterTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        
    def fit(self, text):
        """Training data se vocabulary banata hai"""
        chars = sorted(list(set(text)))  # unique characters
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        print(f"Vocabulary size: {len(self.char_to_id)}")
        
    def encode(self, text):
        """Text ko numbers mein convert karta hai"""
        return [self.char_to_id[ch] for ch in text]
    
    def decode(self, ids):
        """Numbers ko wapas text mein laata hai"""
        return ''.join([self.id_to_char[i] for i in ids])


# ------- Testing -------
text = "hello world! main LLM seekh raha hoon."

tokenizer = CharacterTokenizer()
tokenizer.fit(text)

encoded = tokenizer.encode("hello")
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)