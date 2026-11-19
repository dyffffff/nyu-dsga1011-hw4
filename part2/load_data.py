import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.data_folder = data_folder
        self.split = split

        #load the t5 tokenizer
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

        #process and store the token IDS
        self.encoder_ids, self.sql_ids =self.process_data(data_folder,split,tokenizer)

    def __len__(self):
        return len(self.encoder_ids)


    def process_data(self, data_folder, split, tokenizer):
        # TODO(there are train, dev test split, we need train and dev here)
        #read natural-language inputs
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_texts = load_lines(nl_path)

        #read SQL targets
        if split != 'test':
            sql_path =os.path.join(data_folder,f'{split}.sql')
            sql_texts = load_lines(sql_path)

        else:
            sql_texts = None

        # Tokenize natural-language inputs.
        nl_encoding = tokenizer(
            nl_texts,
            padding = False, #保留各条文本自己的长度，之后组 batch 时再补齐
            truncation = False, #保留完整文本
            add_special_tokens = True, #添加 T5 的结束标记 </s>
        )

        encoder_ids =  nl_encoding['input_ids' ]

        #tokenize sql inputs
        if sql_texts is not None:
            sql_encoding =tokenizer(
                sql_texts,
                padding = False,
                truncation = False,
                add_special_tokens = True, # keep </s> in the sql, id is 1
            )
            sql_ids = sql_encoding['input_ids']

        else:
            sql_ids = None

        return encoder_ids, sql_ids
    

    

        


    def __getitem__(self, idx):
        # TODO
        encoder = torch.tensor(self.encoder_ids[idx],dtype = torch.long)
        if self.sql_ids is None:
            return encoder
        else:
            sql = torch.tensor(self.sql_ids[idx],dtype = torch.long)

            return encoder,sql

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    '''normal_collate_fn(batch)，
    把多条样本组成一个 batch。它收到的内容类似：
    batch = [
    (encoder_0, sql_0),
    (encoder_1, sql_1),
]
    '''
    # TODO
    # Collect input and target sequences.
    encoder_sequences = [item[0] for item in batch]
    sql_sequences = [item[1] for item in batch]
    #Pad inputs to the longest input sequence in this batch.
    encoder_ids = pad_sequence(
        encoder_sequences,
        batch_first = True,
        padding_value = PAD_IDX,
    )
    #mark real tokens as 1 and padding as 0
    encoder_mask = (encoder_ids!=PAD_IDX).long()

    #get decoded sql target, keep </s>
    decoder_targets = pad_sequence(
        sql_sequences,
        batch_first = True,
        padding_value = PAD_IDX,
    )
    #sql_mask = (decoder_targets!=PAD_IDX).long()


    # Initialize decoder inputs with the start/padding token.
    #先创建形状、类型都与 targets 相同，内容全是 0 的 tensor：
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)

    ## Shift target tokens one position to the right.
    #左边表示“所有样本，从第 1 列开始”。
    #右边应该取 decoder_targets 的所有样本、除最后一列以外的内容
    #targets:  [A, B, C, 1]
    #inputs:   [0, A, B, C]
    decoder_inputs[:,1:] = decoder_targets[:,:-1]

    initial_decoder_inputs = decoder_inputs[:,0:1]


    return (encoder_ids, 
            encoder_mask, 
            decoder_inputs,
            decoder_targets,
            initial_decoder_inputs,)

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seq = [item for item in batch]
    encoder_ids = pad_sequence(
        encoder_seq,
        batch_first = True,
        padding_value = PAD_IDX,
    )
    encoder_mask = (encoder_ids!=PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(encoder_seq),1),
                                        PAD_IDX,
                                        dtype = torch.long,)
    return (encoder_ids,encoder_mask,initial_decoder_inputs)

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    """Load raw text for the train, dev, and test prompting splits."""
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
