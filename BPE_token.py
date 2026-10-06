from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    # --------------------------------------------------
    # 1. Convert text into initial character tokens
    # --------------------------------------------------
    def get_word_tokens(self, text):
        words = text.split()

        tokenized_words = []

        for word in words:
            tokens = list(word) + ["</w>"]
            tokenized_words.append(tokens)

        return tokenized_words

    # --------------------------------------------------
    # 2. Count frequency of adjacent token pairs
    # --------------------------------------------------
    def get_pair_counts(self, tokenized_words):

        pair_counts = Counter()

        for tokens in tokenized_words:

            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])

                pair_counts[pair] += 1

        return pair_counts

    # --------------------------------------------------
    # 3. Merge the most frequent pair
    # --------------------------------------------------
    def merge_pair(self, tokenized_words, pair):

        new_tokenized_words = []

        first, second = pair

        merged_token = first + second

        for tokens in tokenized_words:

            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    new_tokens.append(merged_token)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_tokenized_words.append(new_tokens)

        return new_tokenized_words

    # --------------------------------------------------
    # 4. Train BPE
    # --------------------------------------------------
    def fit(self, text):

        tokenized_words = self.get_word_tokens(text)

        print("Initial tokens:")
        print(tokenized_words)

        while True:

            pair_counts = self.get_pair_counts(tokenized_words)

            if not pair_counts:
                break

            # Most frequent pair
            best_pair, frequency = pair_counts.most_common(1)[0]

            # Stop when vocabulary size is reached
            if len(self.vocab) >= self.vocab_size:
                break

            print(
                f"Merging {best_pair} "
                f"(frequency = {frequency})"
            )

            tokenized_words = self.merge_pair(
                tokenized_words,
                best_pair
            )

            # Save merge rule
            self.merges[best_pair] = (
                best_pair[0] + best_pair[1]
            )

            # Update vocabulary
            self.build_vocab(tokenized_words)

        print("\nFinal vocabulary:")
        print(self.vocab)

    # --------------------------------------------------
    # 5. Build vocabulary
    # --------------------------------------------------
    def build_vocab(self, tokenized_words):

        all_tokens = []

        for tokens in tokenized_words:
            all_tokens.extend(tokens)

        unique_tokens = sorted(set(all_tokens))

        self.vocab = {
            token: i
            for i, token in enumerate(unique_tokens)
        }

    # --------------------------------------------------
    # 6. Encode text
    # --------------------------------------------------
    def encode(self, text):

        words = text.split()

        encoded = []

        for word in words:

            tokens = list(word) + ["</w>"]

            changed = True

            while changed:

                changed = False

                for pair, merged_token in self.merges.items():

                    new_tokens = []
                    i = 0

                    while i < len(tokens):

                        if (
                            i < len(tokens) - 1
                            and tokens[i] == pair[0]
                            and tokens[i + 1] == pair[1]
                        ):
                            new_tokens.append(merged_token)
                            i += 2
                            changed = True

                        else:
                            new_tokens.append(tokens[i])
                            i += 1

                    tokens = new_tokens

            for token in tokens:

                if token in self.vocab:
                    encoded.append(self.vocab[token])

        return encoded

    # --------------------------------------------------
    # 7. Decode token IDs back into text
    # --------------------------------------------------
    def decode(self, ids):

        id_to_token = {
            i: token
            for token, i in self.vocab.items()
        }

        tokens = [
            id_to_token[i]
            for i in ids
        ]

        text = ""

        for token in tokens:

            if token == "</w>":
                text += " "

            else:
                text += token

        return text.strip()


# ======================================================
# TESTING
# ======================================================

text = """
low low low lowest lowest
newer newer newer wider wider
"""

tokenizer = BPETokenizer(vocab_size=30)

tokenizer.fit(text)

encoded = tokenizer.encode("lowest")

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)