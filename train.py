import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader , random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import bulingual_Dataset, causal_mask

def get_all_sentences(ds, lang):
    """Extract all sentences for a given language from the dataset."""
    for item in ds['train']:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path(tokenizer_path):
        tokenizer= Tokenizer(WordLevel(unk_token='[unk]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer  = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[EOS]","[SOS]",""],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}.{config["lang_tgt"]}',split="train")

    #build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keap 90% for training and 10% for validation

    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = int(0.1* len(ds_raw))
    train_ds_raw , val_ds_raw  = random_split (ds_raw,[train_ds_size, val_ds_size])

    train_ds = bulingual_Dataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config["lang_tgt"], config['seq_len'])
    val_ds = bulingual_Dataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config["lang_tgt"], config['seq_len'])

    max_len_src = 0
    max_len_tgt =0

    for item in ds_row:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt =max(max_len_tgt, len(tgt_ids)) 
    
    print(f'Max length of source sentence:{max_len_src}')
    print(f'Max length of target sentence:{max_len_tgt}')