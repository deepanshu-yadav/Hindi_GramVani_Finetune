import sentencepiece as spm
import json
import os
from glob import glob

# Create output directory
os.makedirs('tokenizer_output', exist_ok=True)

# Extract texts from manifest
texts = []
with open('train_manifest.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        if 'text' in data and data['text'].strip():
            texts.append(data['text'])

print(f"Found {len(texts)} texts for training")

# Save texts to document.txt (raw corpus)
document_file = 'tokenizer_output/document.txt'
with open(document_file, 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text + '\n')
print(f"Saved raw text corpus to {document_file}")

# Train SentencePiece model
model_prefix = 'tokenizer_output/tokenizer'
spm.SentencePieceTrainer.train(
    input=document_file,  # Now using document.txt directly
    model_prefix=model_prefix,
    vocab_size=1024,
    model_type='bpe',
    character_coverage=0.9995,
    normalization_rule_name='identity',
    remove_extra_whitespaces=False,
    max_sentence_length=4192,
    shuffle_input_sentence=True
)

print(f"Tokenizer saved as {model_prefix}.model and {model_prefix}.vocab")

# Create human-readable vocab.txt
vocab_file = 'tokenizer_output/vocab.txt'
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

with open(vocab_file, 'w', encoding='utf-8') as f:
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        f.write(f"{piece}\n")
print(f"Saved human-readable vocabulary to {vocab_file}")

# Test the tokenizer
test_text = texts[0]
encoded = sp.encode_as_pieces(test_text)
print(f"\nTest encoding:")
print(f"Original: {test_text[:100]}...")
print(f"Encoded: {encoded[:20]}...")

# Optional: Clean up temporary files (comment out if you want to keep them)
# os.remove(document_file)  # Uncomment if you want to remove the corpus after training