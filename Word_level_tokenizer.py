class WordTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        
    def fit(self, text):
        words = text.lower().split()  # simple splitting
        unique_words = sorted(list(set(words)))
        
        # Special tokens add kar rahe hain
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        vocab = special_tokens + unique_words
        
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        print(f"Vocabulary size: {len(self.word_to_id)}")
        
    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id["<UNK>"]) for word in words]
    
    def decode(self, ids):
        return ' '.join([self.id_to_word[i] for i in ids])


# ------- Testing -------
text = "hello world main LLM seekh raha hoon hello"

tokenizer = WordTokenizer()
tokenizer.fit(text)

encoded = tokenizer.encode("hello world seekh raha")
print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded))