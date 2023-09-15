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



    
