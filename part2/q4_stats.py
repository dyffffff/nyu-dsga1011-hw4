from pathlib import Path
from transformers import T5TokenizerFast

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_lines(path):
    lines = path.read_text(encoding="utf-8").splitlines()

    return lines

#Load train.nl, train.sql, dev.nl, and dev.sql.
train_lines = load_lines(DATA_DIR/'train.nl')
train_sql_lines =load_lines(DATA_DIR/'train.sql')
dev_lines = load_lines(DATA_DIR/'dev.nl')
dev_sql_lines = load_lines(DATA_DIR/'dev.sql')

#print ethe number of examples in each file
print(f'the number of examples in train_lines: {len(train_lines)}')
print(f'the number of examples in train_sql: {len(train_sql_lines)}')
print(f'the number of examples in dev_lines: {len(dev_lines)}')
print(f'the number of examples in dev_sql: {len(dev_sql_lines)}')


tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

# Inspect one natural-language example.
text = train_lines[0]
encoded = tokenizer(
    text,
    add_special_tokens = False,
    padding = False,
    truncation = False,

)

token_ids = encoded['input_ids']

print("Text:", text)
print('Tokens:', tokenizer.convert_ids_to_tokens(token_ids))
print('Token IDs:', token_ids)
print('Token count:',len(token_ids))


# 
def compute_stats(lines, tokenizer):
    encoded = tokenizer(
        lines,
        add_special_tokens = False,
        padding = False,
        truncation = False,

    )

    token_sequences = encoded["input_ids"]

    '''
    token_sequences is something like this
    [
    [570, 66, 8, ...],  # First example
    [839, 140, ...],    # Second example
]
    '''

    #Count the examples.
    num_examples = len(token_sequences)

    #Calculate the mean token sequence length.
    lengths = [len(line) for line in token_sequences]
    mean_seq_length = sum(lengths)/num_examples

    # Count distinct token IDs across all sequences.
    vocab = set()
    for line in token_sequences:
        vocab.update(line)

    vocab_size = len(vocab)

    return {
        'num_examples':num_examples,
        'mean_length': mean_seq_length,
        'vocab_size' : vocab_size,
    }


print("Train NL:", compute_stats(train_lines, tokenizer))
print("Train SQL:", compute_stats(train_sql_lines, tokenizer))
print("Dev NL:", compute_stats(dev_lines, tokenizer))
print("Dev SQL:", compute_stats(dev_sql_lines, tokenizer))

