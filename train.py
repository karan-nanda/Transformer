from model import build_transformer
 

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

#HuggingFace imports
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src,tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id(['[EOS]'])
    
    #Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source,source_mask)
    #Initialize the decorder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        #Build the mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)
        
        #calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input,decoder_mask)
        
        #Get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim =1
            )
        
        if next_word ==eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step,writer, num_examples = 2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    try:
        #Get the console window
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
            
    except:
        #Use 80 as default if no console width
        console_width = 80
        
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, targer and the model output
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':> 12}{source_text}")
            print_msg(f"{f'TARGET: ':> 12}{target_text}")
            print_msg(f"{f'PREDICTED: ':> 12}{model_out_text}")
            
            if count == num_examples:
                print_msg('-'*console_width)
                break
            
    if writer:
        #Eval the character error rate
        #Compute the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        
        #Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config,ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer= Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer
def get_ds(config):
    #Divide the split into two(Training, Testing)
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    #Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    #Keep 90% for training, 10% for the validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    #Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt(item['translation'][config['lang_tgt']]).ids
        
