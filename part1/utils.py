import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.

def introduce_typo(word):
    if len(word) < 4 or not word.isalpha():
        return word

    chars = list(word)

    #randomly choose the index of letter to be swapped, excluding first and last leeter
    index = random.randint(1,len(chars)-3) 
    chars_new = chars.copy()
    #swap two letters
    chars_new[index+1] = chars[index]
    chars_new[index] = chars[index+1]
    #join letter together
    word_trans = "".join(chars_new)

    return word_trans

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    p = 0.4
    parts = re.split(r"(\W+)", example['text'])

    for i, part in enumerate(parts):
        if len(part) < 4 or not part.isalpha():
            continue
        if random.random()<=p:
            parts[i] = introduce_typo(part)

    example['text'] = ''.join(parts)




    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    

    ##### YOUR CODE ENDS HERE ######

    return example
