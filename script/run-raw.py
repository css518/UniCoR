import argparse
import logging
from math import e
import os
import pickle
from numpy.random import f
import torch
import json
import re
import numpy as np
import random
from itertools import cycle

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                  RobertaModel, RobertaConfig, RobertaTokenizer, T5Config, T5Tokenizer, T5ForConditionalGeneration)

logger = logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing
cpu_cont = 16

from util import set_seed, load_json, remove_comments_and_docstrings,remove_language, json_pretty_dump, print_result, save_json_data, Loss_Curve
from model import UniCoR

LABEL = {
        'python': 0,
        'java': 1,
        'ruby': 2,
        'go': 3,
        'php': 4,
        'javascript': 5
    }

ruby_special_token = ['keyword', 'identifier', 'separators', 'simple_symbol', 'constant', 'instance_variable',
 'operator', 'string_content', 'integer', 'escape_sequence', 'comment', 'hash_key_symbol',
  'global_variable', 'heredoc_beginning', 'heredoc_content', 'heredoc_end', 'class_variable',]

java_special_token = ['keyword', 'identifier', 'type_identifier',  'separators', 'operator', 'decimal_integer_literal',
 'void_type', 'string_literal', 'decimal_floating_point_literal', 
 'boolean_type', 'null_literal', 'comment', 'hex_integer_literal', 'character_literal']

go_special_token = ['keyword', 'identifier', 'separators', 'type_identifier', 'int_literal', 'operator', 
'field_identifier', 'package_identifier', 'comment',  'escape_sequence', 'raw_string_literal',
'rune_literal', 'label_name', 'float_literal']

javascript_special_token =['keyword', 'separators', 'identifier', 'property_identifier', 'operator', 
'number', 'string_fragment', 'comment', 'regex_pattern', 'shorthand_property_identifier_pattern', 
'shorthand_property_identifier', 'regex_flags', 'escape_sequence', 'statement_identifier']

php_special_token =['text', 'php_tag', 'name', 'operator', 'keyword', 'string', 'integer', 'separators', 'comment', 
'escape_sequence', 'ERROR',  'boolean', 'namespace', 'class', 'extends']

python_special_token =['keyword', 'identifier', 'separators', 'operator', '"', 'integer', 
'comment', 'none', 'escape_sequence']


special_token={
    'python':python_special_token,
    'java':java_special_token,
    'ruby':ruby_special_token,
    'go':go_special_token,
    'php':php_special_token,
    'javascript':javascript_special_token
}

all_special_token = []
for key, value in special_token.items():
    all_special_token = list(set(all_special_token ).union(set(value)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int,
                      help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                      help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                      help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
    parser.add_argument('--gpu_per_node', default=2, type=int,
                      help='number of gpus per node')


    parser.add_argument("--local-rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")  
        
    #model
    parser.add_argument('--do_inter_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_aug_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_inner_code_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_inner_nl_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_mmd_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_kl_loss', action='store_true', help='debug mode', required=False)
    parser.add_argument('--graph_hidden_dim', type=int, default=1024, required=False)
    parser.add_argument('--encoder_hidden_dim', type=int, default=768, required=False)
    parser.add_argument('--classifier_hidden_size', type=int, default=256, required=False)
    parser.add_argument('--classification', type=int, default=3, required=False)

    parser.add_argument("--cls", action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_whitening', action='store_true', help='do_whitening https://github.com/Jun-jie-Huang/WhiteningBERT', required=False)
    parser.add_argument("--time_score", default=1, type=int,help="cosine value * time_score")   
    parser.add_argument("--dropout", default=0.1, type=float, required=False, help="dropout")


    # dataset config
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
        
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    #training config
    parser.add_argument("--mlm_probability", default=0.1, type=float, required=False)
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay',default=0.01, type=float,required=False)

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience", default=5, type=int,
                        help="Early Stop")
    parser.add_argument("--max_steps", default=100000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Log every X updates steps.")

    # moco
    # moco specific configs:
    parser.add_argument('--moco_dim', default=768, type=int,
                        help='feature dimension (default: 768)')
    parser.add_argument('--moco_k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536), which is divided by 32, etc.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', action='store_true',help='use mlp head')

    ## Required data file parameters
    parser.add_argument("--dataset", default="CSN", type=str, required=True,
                    help="The input data file (a json file).")
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    
    parser.add_argument("--model", default="Unixcoder", type=str,
                        help="The model name.")
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
        
    #print arguments
    args = parser.parse_args()
    return  args

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_ids,
                 query_ids,
                 comment_ids,
                 index,
                 label, 
                 language):
            self.code_ids = code_ids
            self.query_ids = query_ids
            self.comment_ids = comment_ids
            self.index = index
            self.label = label
            self.language = LABEL[language.lower()]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps({
            "code_ids": self.code_ids,
            "query_ids": self.query_ids,
            "comment_ids": self.comment_ids,
            "index": self.index,
            "label": self.label,
            "language": self.language,
        }, indent=2)

    def __repr__(self):
        return str(self.to_json_string())      

def convert_examples_to_features(js,tokenizer,args):
    if 'comment' not in js:
        logger.info('wrong index: {}'.format(js['index']))
        return None

    try:
        code = " ".join(remove_comments_and_docstrings(js['func'], js['language']).split())

        code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length

        comment_tokens = tokenizer.tokenize(remove_language(js['comment'], js['language']))[:args.nl_length-4]
        comment_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+ comment_tokens +[tokenizer.sep_token]
        comment_ids = tokenizer.convert_tokens_to_ids(comment_tokens)
        padding_length = args.nl_length - len(comment_ids)
        comment_ids += [tokenizer.pad_token_id]*padding_length

        query_tokens = tokenizer.tokenize(remove_language(js['query'], js['language']))[:args.nl_length-4]
        query_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+ query_tokens +[tokenizer.sep_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        padding_length = args.nl_length - len(query_ids)
        query_ids += [tokenizer.pad_token_id]*padding_length

        return InputFeatures(code_ids, comment_ids, query_ids, js["index"], js['label'], js['language'])
    
    except Exception as e:
        logger.info('wrong index {} {}'.format(js["index"], e))
        return None

class MyDataset(Dataset):
    def __init__(self, tokenizer, args, file=None, datapair=None, pooler=None, istrain=True, iscodebase=False):
        self.data = datapair
        self.istrain = istrain
        self.iscodebase = iscodebase

        file_name = re.split(r'[2_]', file)
        self.name = file_name
        cache_file1 = args.output_dir+'/'+ '{}.pkl'.format(file_name[0])
        cache_file2 = args.output_dir+'/'+ '{}.pkl'.format(file_name[1])

        for cache_file,file_part in zip([cache_file1, cache_file2], [file_name[0], file_name[1]]):
            if not os.path.exists(cache_file):
                data, save = [], []
                data = load_json('./dataset/train/{}_with_nl_ready.jsonl'.format(args.dataset, file_part))
                for js in tqdm(data):
                    save.append(convert_examples_to_features(js,tokenizer,args))
                save = [item for item in save if item is not None]
                save = {item.index : item for item in save}
                pickle.dump(save, open(cache_file,'wb'))

        self.examples1=pickle.load(open(cache_file1,'rb'))
        self.examples2=pickle.load(open(cache_file2,'rb'))
        if not self.istrain:
            query_index = list(set([_item.split('-')[0] for _item in self.data]))
            self.examples1 = [values for index, values in self.examples1.items() if index in query_index]
            self.examples2 = [values for _, values in self.examples2.items()]
        
        if args.debug:
            self.data = self.data[:args.n_debug_samples]

    def __len__(self):
        if not self.istrain:
            if self.iscodebase:
                return len(self.examples2)
            else:
                return len(self.examples1)
        else:
            return len(self.data)
    
    def __getitem__(self, i):
        # logger.info(self.name)
        if not self.istrain:
            if self.iscodebase:
                data = self.examples2[i]
            else:
                data = self.examples1[i]
            return (torch.tensor(data.code_ids),torch.tensor(data.query_ids), torch.tensor(data.comment_ids), data.index, data.label)
        else:
            data_index = self.data[i].split('-')
            data1 = self.examples1[data_index[0]]
            data2 = self.examples2[data_index[1]]
            if i % 2 == 0:
                return (torch.tensor(data1.code_ids),torch.tensor(data1.query_ids), torch.tensor(data1.comment_ids), torch.tensor(data1.language),
                    torch.tensor(data2.code_ids),torch.tensor(data2.query_ids), torch.tensor(data2.comment_ids), torch.tensor(data2.language))
            else:
                return (torch.tensor(data1.code_ids), torch.tensor(data1.comment_ids),torch.tensor(data1.query_ids), torch.tensor(data1.language),
                    torch.tensor(data2.code_ids),torch.tensor(data2.comment_ids), torch.tensor(data2.query_ids), torch.tensor(data2.language))

def mask_tokens(inputs, replaces, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def replace_with_type_tokens(inputs, replaces, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = replaces[indices_replaced]

    return inputs, labels

def replace_special_token_with_type_tokens(inputs, replaces, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =  speical_token_ids

    return inputs, labels

def replace_special_token_with_mask(inputs, replaces, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =tokenizer.convert_tokens_to_ids(tokenizer.mask_token) 

    return inputs, labels

def train(args, model, tokenizer, pool):
    traindata = load_json(args.train_data_file)[0]
    train_datasets = []
    for name, values in traindata.items():
        train_datasets.append(MyDataset(tokenizer, args, name, values, pool, istrain=True, iscodebase=False))
    
    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,drop_last=True)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]

    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,num_training_steps=args.max_steps)

    training_data_step = sum ([len(item)//args.train_batch_size for item in train_datasets])
    logger.info("***** Running training *****")
    logger.info("  Num step = %d", training_data_step)
    logger.info("  Num queue = %d", args.moco_k)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)

    best_, step, patience = {item:(-1, 0) for item in ['code2code_MRR', 'nl2code_MRR', 'code2code_MAP', 'nl2code_MAP']}, 0, 0
    loss_record = Loss_Curve(['Loss', 'Contrastive Loss','MMD Loss', 'KL Loss'])

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    last_state_file = os.path.join(checkpoint_last, 'last_state.pt')

    if os.path.exists(checkpoint_last):
        last_state = torch.load(last_state_file)
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-best-code2code_MRR/model.bin')), strict=False)
        model.load_state_dict(last_state['model'], strict=False)
        scheduler.load_state_dict(last_state['scheduler'])
        optimizer.load_state_dict(last_state['optimzier'])
        step = int(last_state['step'])
        logger.info("Loaded model state from %s", checkpoint_last)
        logger.info("Loaded global step: %d", step)
    # else:
    #     results = evaluate(args, model, tokenizer, args.eval_data_file, pool)
    #     logger.info("***** Eval valid results *****")
    #     logger.info(print_result(results))


    model.zero_grad()
    model.train()

    set_seed(args.seed)
    probs=[len(x) for x in train_datasets]
    probs=[x/sum(probs) for x in probs]
    probs=[x**0.7 for x in probs]
    probs=[x/sum(probs) for x in probs]

    special_token_list = all_special_token 
    special_token_id_list = tokenizer.convert_tokens_to_ids(special_token_list)

    code_transform = [mask_tokens, replace_with_type_tokens, replace_special_token_with_type_tokens, replace_special_token_with_mask]

    while True:
        train_dataloader=np.random.choice(train_dataloaders, 1, p=probs)[0]
        step+=1
        batch=next(train_dataloader)
        model.train()

        # p = float(step) / args.max_steps  
        # alpha = 2. / (1. + np.exp(-100 * p)) - 1

        code1_inputs = batch[0].to(args.device)
        nl1_inputs = batch[1].to(args.device)
        nl1_aug_inputs = batch[2].to(args.device)
        code2_inputs = batch[4].to(args.device)
        nl2_inputs = batch[5].to(args.device)
        nl2_aug_inputs = batch[6].to(args.device)

        random.seed(step)        
        code1_aug_ids = code1_inputs.clone()
        code2_aug_ids = code2_inputs.clone()

        code1_aug_ids[:, 3:], _ = code_transform[step % 4](code1_inputs.clone()[:, 3:], code1_inputs.clone()[:, 3:], random.choice(special_token_id_list), tokenizer, args.mlm_probability)
        code2_aug_ids[:, 3:], _ = code_transform[step % 4](code2_inputs.clone()[:, 3:], code2_inputs.clone()[:, 3:], random.choice(special_token_id_list), tokenizer, args.mlm_probability)
        if step % 4 == 0:
            nl1_aug_ids = nl1_inputs.clone()
            nl2_aug_ids = nl2_inputs.clone()
            nl1_aug_ids[:, 3:], _ = mask_tokens(nl1_inputs.clone()[:, 3:], nl1_inputs.clone()[:, 3:], random.choice(special_token_id_list), tokenizer, args.mlm_probability)
            nl2_aug_ids[:, 3:], _ = mask_tokens(nl2_inputs.clone()[:, 3:], nl2_inputs.clone()[:, 3:], random.choice(special_token_id_list), tokenizer, args.mlm_probability)
        else:
            nl1_aug_ids = nl1_aug_inputs
            nl2_aug_ids = nl2_aug_inputs

        loss_cl, loss_mmd, loss_kl= model(code1_q_r=code1_inputs, code1_k_r=code1_aug_ids, code2_q_r=code2_inputs, code2_k_r=code2_aug_ids, nl1_q_r=nl1_inputs, nl1_k_r=nl1_aug_ids, nl2_q_r=nl2_inputs, nl2_k_r=nl2_aug_ids)

        loss = loss_cl
    
        if args.do_mmd_loss:
            loss += loss_mmd

        if args.do_kl_loss:
            loss += loss_kl

        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.local_rank != -1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / torch.distributed.get_world_size()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        loss_record.add((loss, loss_cl, loss_mmd, loss_kl))

        if args.local_rank in [-1, 0]:
            if step % 100 == 0:
                logger.info(loss_record.get_loss_info(step))

            if args.save_steps > 0 and step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                encoder_to_save = model.module.code_encoder_q  if hasattr(model,'module') else model.code_encoder_q
                model_to_save = model.module if hasattr(model,'module') else model

                   # Take care of distributed/parallel training
                encoder_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)

                last_state = {
                    'model': model_to_save.state_dict(),
                    'optimzier': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': str(step)
                }
                torch.save(last_state, os.path.join(last_output_dir, "last_state.pt"))


                results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
                for _key, _item in results.items():
                    if _key not in best_:
                        best_[_key]=(-1,0)
                    if _item >= best_[_key][0]:
                        best_[_key] = (_item, step)
                
                flag = False
                for key in ['code2code_MRR', 'nl2code_MRR', 'code2code_MAP', 'nl2code_MAP']:
                    sum_result = [ _item for _key, _item in results.items() if key in _key]
                    sum_result = round(sum(sum_result) / len(sum_result), 6)

                    if sum_result >= best_[key][0]:
                        best_[key] = (sum_result, step)
                        flag = True
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(key))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving best model checkpoint to %s...", output_dir)
                    
                    logger.info(f'step: {step} \t cur_{key}: {sum_result} \t best_step:{best_[key][1]} \t best_{key}: {best_[key][0]}')

                if flag:
                    patience = 0
                else:
                    patience += 1
                logger.info(f"Patience: {patience}")

                logger.info(print_result(best_))
                loss_record.draw_graph(args.output_dir)
                logger.info("  "+"*"*20)
                
            if (args.max_steps > 0 and step > args.max_steps) or (args.patience > 0 and patience >= args.patience):
                logger.info("Training is Early Stop")
                break

def do_evaluate(args, model, dataloader):
    # query_code_vecs, query_nl_vecs, query_uids, query_labels
    model.eval()
    code_vecs, nl_vecs, uids, labels = [], [], [], []

    for batch in dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)

        code_vec = model.do_repre(model.code_encoder_q, code_inputs)
        nl_vec = model.do_repre(model.nl_encoder_q, nl_inputs)

        code_vecs.append(code_vec.cpu().detach().numpy())
        nl_vecs.append(nl_vec.cpu().detach().numpy())

        uids.append(batch[3])
        labels.append(batch[4])
    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    uids = np.concatenate(uids,0)
    labels = np.concatenate(labels,0)

    return code_vecs, nl_vecs, uids, labels

def do_metric(query_vecs, key_vecs, query_uids, key_uids, query_idx, key_idx, prefix):
    key_idx = key_idx.tolist()
    url_to_indices = {}
    for idx, url in enumerate(key_uids):
        if url not in url_to_indices:
            url_to_indices[url] = []
        url_to_indices[url].append(idx)

    scores=np.matmul(query_vecs,key_vecs.T)
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    ranks, MAP = [], []
    for url, index, sort_id in zip(query_uids, query_idx, sort_ids):
        if url not in url_to_indices:
            continue
        
        indices = url_to_indices[url]
        
        sort_id = sort_id.tolist()
        indices_rank = [(sort_id.index(_item) + 1, _item) for _item in indices]
        indices_rank.sort(key=lambda x: x[0])

        if index in key_idx:
            indices_rank_noself = []
            bad_index = key_idx.index(index)
            flag = False
            for _item in indices_rank:
                if _item[1] == bad_index:
                    flag = True
                else:
                    if flag:
                        indices_rank_noself.append((_item[0] - 1, _item[1]))
                    else:
                        indices_rank_noself.append((_item[0], _item))
            indices_rank =  indices_rank_noself           

        
        if len(indices_rank) == 0:
            continue

        indices_rank = [_item[0] for _item in indices_rank]

        ranks.append(1/indices_rank[0])
        indices_rank = [(_idx + 1) / (_idy) for _idx, _idy in enumerate(indices_rank)]
        MAP.append(np.mean(indices_rank))
    
    result = {
        '{}_MAP'.format(prefix): round(float(np.mean(MAP)),6),
        '{}_MRR'.format(prefix): round(float(np.mean(ranks)),6),
    }
    return result

def evaluate(args, model, tokenizer, file_name, pool, eval_when_training=False):

    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    model_eval = model_eval.to(args.device)

    result_all, codebase = {}, {}
    evaldata = load_json(file_name)[0]

    logger.info("***** Running evaluation *****")

    for name, values in evaldata.items():
        query_dataset = MyDataset(tokenizer, args, name, values, pool, istrain=False, iscodebase=False)
        query_sampler = SequentialSampler(query_dataset)
        query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
        
        if query_dataset.name[1] not in codebase:
            code_dataset = MyDataset(tokenizer, args, name, values, pool, istrain=False, iscodebase=True)
            code_sampler = SequentialSampler(code_dataset)
            code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)

        with torch.no_grad():
            query_code_vecs, query_nl_vecs, query_uids, query_labels = do_evaluate(args, model_eval, query_dataloader)
            if query_dataset.name[1] not in codebase:
                codebase[query_dataset.name[1]] = do_evaluate(args, model_eval, code_dataloader)

            base_code_vecs, _, base_uids, base_labels = codebase[query_dataset.name[1]]

        result_code_nl = do_metric(query_nl_vecs, base_code_vecs, query_labels, base_labels, query_uids, base_uids, '{} nl2code'.format(name))
        result_code_code = do_metric(query_code_vecs, base_code_vecs, query_labels, base_labels, query_uids, base_uids, '{} code2code'.format(name))
        result = {**result_code_nl,**result_code_code}

        result_all.update(result)

        if args.debug:
            break

    return result_all

def main():
    args = parse_args()
        #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # Set seed
    set_seed(args.seed)
    args.train_batch_size = args.train_batch_size * args.n_gpu
    # args.moco_k = args.train_batch_size * 60
    json_pretty_dump(args, os.path.join(args.output_dir, "params.json"))

    args.gpu_per_node = args.n_gpu
    args.device = device
    torch.cuda.empty_cache()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    if args.local_rank not in [-1, 0]:
        logger.disabled = True

    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    logger.info("gpu_per_node: %s, rank: %s, local_rank: %s, world_size: %s", args.gpu_per_node, args.rank, args.local_rank, args.world_size)
    logger.info("Parameters %s", args)
    
    pool = multiprocessing.Pool(cpu_cont)

    if "codet5" in args.model_name_or_path:
        config = T5Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer =  T5Tokenizer.from_pretrained(args.tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = model.encoder
    else:
        config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = RobertaModel.from_pretrained(args.model_name_or_path)

    special_tokens_dict = {'additional_special_tokens': all_special_token}
    logger.info(" new token %s"%(str(special_tokens_dict)))
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model = UniCoR(model, args)
    model.to(args.device)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl' if args.n_gpu > 1 else 'gloo',
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size * args.gpu_per_node,
            rank=args.rank * args.gpu_per_node + args.local_rank
        )

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)   

    if args.n_gpu > 1 and args.local_rank == -1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args, model, tokenizer, pool)

if __name__ == "__main__":
    main()