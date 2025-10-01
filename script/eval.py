# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
running the evaluation script to evaluate the model on the test set and obtain MRR and MAP scores.
'''

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import logging
import os
import re
import pickle
import random
import torch
import json
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional, Union
from tqdm import tqdm
from model import Model_Eval as Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler

logger = logging.getLogger(__name__)


# Add special tokens follows UnixCoder.
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

class InputFeatures(object):
    """A single training/test features for an example.
    
    Attributes:
        code_tokens: Tokenized code
        code_ids: Token IDs of the code
        nl_tokens: Tokenized natural language
        nl_ids: Token IDs of the natural language
        ids: Combined token IDs
        url: URL or identifier for the example
    """
    def __init__(self,
                 code_tokens: List[str],
                 code_ids: List[int],
                 nl_tokens: List[str],
                 nl_ids: List[int],
                 ids: List[int],
                 url: str,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.ids = ids
        self.url = url

class InputFeatures_gcb(object):
    """A single training/test features for an example on GraphCodeBert.
    
    Attributes:
        code_tokens: Tokenized code
        code_ids: Token IDs of the code
        position_idx: Position indices
        attn_mask: Attention mask matrix
        nl_ids: Token IDs of the natural language
        ids: Combined token IDs
        index: Index of the example
        url: URL or identifier for the example
    """
    def __init__(self,
                 code_tokens: List[str],
                 code_ids: List[int],
                 position_idx: List[int],
                 attn_mask: np.ndarray,
                 nl_ids: List[int],
                 ids: List[int],
                 index: int,
                 url: str,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.attn_mask=attn_mask
        self.nl_ids = nl_ids
        self.ids = ids
        self.index = index
        self.url=url

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code: str, parser: List[Any], lang: str) -> Tuple[List[str], List[Any]]:
    """Extract dataflow graph from code using parser, this function is used for GraphCodeBert.
    Args:
        code: Code string to process
        parser: Parser and dataflow extraction function
        lang: Programming language of the code
    Returns:
        Tuple containing code tokens and dataflow graph
    """

    try:
        code=remove_comments_and_docstrings(code,lang)     #remove comments
    except:
        pass    

    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

def save_json_data(save_dir: str, filename: str, data: Any) -> None:
    """Save data to a JSON file in the specified directory.
    
    Args:
        save_dir: Directory path to save the file
        filename: Name of the output file
        data: Data to save (usually a dictionary or list)
    """
    os.makedirs(save_dir, exist_ok=True)
        
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        if isinstance(data, list):
            if isinstance(data[0], (str, list, dict)):
                for item in data:
                    f.write(json.dumps(item))
                    f.write('\n')
            else:
                json.dump(data, f)
        elif isinstance(data, dict):
            json.dump(data, f)
        else:
            raise RuntimeError(f'Unsupported type: {type(data)}')
    logger.info(f"saved dataset in {filename}")

        
def convert_examples_to_features(js: Dict[str, Any], tokenizer: Any, args: Any, start: List[str]) -> InputFeatures:
    """Convert example to input features with tokenized code and natural language.
    Args:
        js: Dictionary containing code and query data
        tokenizer: Tokenizer for converting text to token IDs
        args: Arguments containing configuration parameters like sequence lengths
        start: List of starting tokens (typically including CLS token)
    Returns:
        InputFeatures object containing tokenized code, natural language, and combined IDs
    """
    code =  ' '.join(js['func'].split()) if "func" in js else ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens_f = start + code_tokens[:args.code_length-1-len(start)] + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens_f)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['query']) if type(js['query']) is list else ' '.join(js['query'].split())
    nl_tokens = tokenizer.tokenize(nl)
    nl_tokens_f = start + nl_tokens[:args.nl_length-1-len(start)] + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens_f)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    tokens = start + nl_tokens[:args.nl_length-1] + [tokenizer.sep_token] + code_tokens[:args.code_length-1] + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.nl_length + args.code_length + len(start) - len(ids)
    ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code, code_ids, nl_tokens, nl_ids, ids ,js['label'] if "label" in js else js["url"].split('-')[0])

def convert_examples_to_features_bge(js: Dict[str, Any], tokenizer: Any, args: Any) -> InputFeatures:
    """Convert example to input features with tokenized code and natural language for BGE.
    Args:
        js: Dictionary containing code and query data
        tokenizer: Tokenizer for converting text to token IDs
        args: Arguments containing configuration parameters like sequence lengths
    Returns:
        InputFeatures object containing tokenized code, natural language, and combined IDs
    """
    code =  ' '.join(js['func'].split()) if "func" in js else ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['query']) if type(js['query']) is list else ' '.join(js['query'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    tokens = nl_tokens+ code_tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.nl_length + args.code_length - len(ids)
    ids += [tokenizer.pad_token_id]*padding_length
    
    try:
        return InputFeatures(code, code_ids,nl_tokens,nl_ids, ids, js['label'] if "label" in js else js["url"].split('-')[0])
    except:
        logging.info(js)

def convert_examples_to_features_gcb(js: Dict[str, Any], tokenizer: Any, args: Any, lang: str) -> InputFeatures_gcb:
    """Convert example to input features for the GraphCodeBert.
    
    Args:
        js: Dictionary containing code and query data
        tokenizer: Tokenizer for converting text to token IDs
        args: Arguments containing configuration parameters
        lang: Programming language of the code
        
    Returns:
        InputFeatures_gcb object containing tokenized code, position indices, attention mask, and other features
    """
    #code
    parser=parsers[lang]
    #extract data flow
    code = js['func'] if "func" in js else js['code']
    code_tokens,dfg=extract_dataflow(code, parser, lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length_code=args.code_length+args.data_flow_length-len(code_ids)

    position_idx+=[tokenizer.pad_token_id]*padding_length_code
    code_ids+=[tokenizer.pad_token_id]*padding_length_code    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
    #nl

    comment_tokens=tokenizer.tokenize(' '.join(js['query']) if type(js['query']) is list else ' '.join(js['query'].split()))
    comment_tokens =[tokenizer.cls_token]+comment_tokens[:args.nl_length-2]+[tokenizer.sep_token]
    comment_ids =  tokenizer.convert_tokens_to_ids(comment_tokens)
    padding_length = args.nl_length - len(comment_ids)
    comment_ids+=[tokenizer.pad_token_id]*padding_length

    tokens = code_tokens+comment_tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.nl_length + args.code_length + args.data_flow_length - len(ids)
    ids += [tokenizer.pad_token_id]*padding_length

    attn_mask=np.zeros((args.code_length+args.data_flow_length,
                        args.code_length+args.data_flow_length),dtype=bool)
    node_index=sum([i>1 for i in position_idx])
    max_length=sum([i!=1 for i in position_idx])
    attn_mask[:node_index,:node_index]=True
    for idx,i in enumerate(code_ids):
        if i in [0,2]:
            attn_mask[idx,:max_length]=True
    for idx,(a,b) in enumerate(dfg_to_code):
        if a<node_index and b<node_index:
            attn_mask[idx+node_index,a:b]=True
            attn_mask[a:b,idx+node_index]=True
    for idx,nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a+node_index<len(position_idx):
                attn_mask[idx+node_index,a+node_index]=True
    
    return InputFeatures_gcb(code,code_ids,position_idx,attn_mask,comment_ids,ids,js["index"], js['label'] if "label" in js else js["url"].split('-')[0])

class TextDataset(Dataset):
    """Dataset class for text data containing code and natural language examples.
    Attributes:
        examples: List of InputFeatures objects containing tokenized code and natural language
        lang: Programming language of the code examples
    """
    def __init__(self, tokenizer: Any, args: Any, file_path: Optional[str] = None):
        """Initialize the TextDataset with data from the specified file.
        Args:
            tokenizer: Tokenizer for converting text to token IDs
            args: Arguments containing configuration parameters
            file_path: Path to the data file
        """
        self.examples: List[Union[InputFeatures, InputFeatures_gcb]] = []
        self.lang = re.split(r'[/.]', file_path)[-2]

        if args.model.endswith('F'):
            name = args.model[:-1]
        else:
            name = args.model

        cache_file = os.path.join(args.cache_dir, f'dataset-{name}-{args.dataset}-{self.lang}.pkl')

        if os.path.exists(cache_file):
            logger.info(f' Loading dataset from {cache_file} ... ')
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            logger.info(f' Dealing dataset to {cache_file} ... ')
            data = []
            with open(file_path) as f:
                if "jsonl" in file_path:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        if 'function_tokens' in js:
                            js['code_tokens'] = js['function_tokens']
                        if 'code_tokens' in js:
                            js['code'] = " ".join(js['code_tokens'])
                        data.append(js)
                elif "codebase" in file_path or "code_idx_map" in file_path:
                    js = json.load(f)
                    for key in js:
                        temp = {}
                        temp['code_tokens'] = key.split()
                        temp["retrieval_idx"] = js[key]
                        temp['doc'] = ""
                        temp['docstring_tokens'] = ""
                        data.append(temp)
                elif "json" in file_path:
                    for js in json.load(f):
                        data.append(js) 

            for js in tqdm(data):
                if name.lower() in ['']:  # graphcodebert
                    self.examples.append(convert_examples_to_features_gcb(js, tokenizer, args, self.lang))
                elif name.lower() in ['zc3', 'codebert', 'graphcodebert']:
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, [tokenizer.cls_token]))
                elif name.lower() in ['bge', 'roberta']:
                    self.examples.append(convert_examples_to_features_bge(js, tokenizer, args))
                elif name.lower() in ['cocosoda', 'unixcoder'] or 'My' in name:
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]))
                else:
                    logger.info('wrong Model Data')
            
            pickle.dump(self.examples, open(cache_file, 'wb'))
                                 
        
    def __len__(self) -> int:
        """Return the number of examples in the dataset.
        Returns:
            Number of examples in the dataset
        """
        return len(self.examples)

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, ...]:
        """Get the example at the specified index.
        Args:
            _idx: Index of the example to retrieve
        Returns:
            Tuple containing tensors for code IDs, attention mask (if available),
            position indices (if available), natural language IDs, code tokens,
            combined IDs, and URL
        """
        if hasattr(self.examples[_idx], "attn_mask"):
            # For GraphCodeBERT
            return (torch.tensor(self.examples[_idx].code_ids), torch.tensor(self.examples[_idx].attn_mask), torch.tensor(self.examples[_idx].position_idx), torch.tensor(self.examples[_idx].nl_ids), self.examples[_idx].code_tokens, torch.tensor(self.examples[_idx].ids), self.examples[_idx].url)
        else:
            return (torch.tensor(self.examples[_idx].code_ids), torch.tensor(self.examples[_idx].nl_ids), self.examples[_idx].code_tokens, torch.tensor(self.examples[_idx].ids), self.examples[_idx].url)
            

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def do_evaluate(args: Any, model: torch.nn.Module, mode: str, dataloader: torch.utils.data.DataLoader):
    """Evaluate the model on the given data and generate embeddings.
    Args:
        args: Arguments containing configuration parameters
        model: Model to evaluate
        mode: Evaluation mode ('nlp', 'code', 'remix', 'code+code', 'nlp+code')
        dataloader: DataLoader providing batches of data
    Returns:
        Tuple containing embeddings, URLs, and indices
    """
    model.eval()

    model_eval = model.module if hasattr(model,'module') else model
    vecs, urls, idxs = [], [], []

    for batch in dataloader: 
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[-4].to(args.device)

        with torch.no_grad():
            nlvec = model_eval(inputs=nl_inputs)

            if len(batch) > 5:
                attn_mask = batch[1].to(args.device)
                position_idx =batch[2].to(args.device)
                codevec = model_eval(inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            else:
                codevec = model_eval(inputs=code_inputs)

            if mode == 'nlp':
                vec = nlvec
            elif mode == 'code':
                vec = codevec
            elif mode == 'remix':
                remix_inputs = batch[-2].to(args.device)
                vec = model_eval(inputs=remix_inputs)
            elif mode == 'code+code':
                vec = torch.cat([codevec, codevec], dim=-1)
            elif mode == 'nlp+code':
                vec = torch.cat([nlvec, codevec], dim=-1)

            vecs.append(vec.cpu().detach().numpy())
            urls.append(list(batch[-1]))
            idxs.append(list(batch[-3]))
    
    vecs = np.concatenate(vecs,0)
    urls = np.concatenate(urls,0)
    idxs = np.concatenate(idxs,0)
    return vecs, urls, idxs

def do_metric(prefix: str, query_vecs: np.ndarray, key_vecs: np.ndarray, query_uids: np.ndarray, key_uids: np.ndarray, query_idx: Optional[np.ndarray] = None, key_idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute evaluation metrics like MAP (Mean Average Precision) and MRR (Mean Reciprocal Rank).
    Args:
        prefix: Prefix for metric names in the result dictionary
        query_vecs: Embeddings for the query examples
        key_vecs: Embeddings for the candidate examples
        query_uids: Unique identifiers for the query examples
        key_uids: Unique identifiers for the candidate examples
        query_idx: Optional indices for the query examples
        key_idx: Optional indices for the candidate examples
    Returns:
        Dictionary containing MAP and MRR scores with the specified prefix
    """
    scores=np.matmul(query_vecs,key_vecs.T)
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    MAP, MRR = [], []
    for label, idx, sort_id in zip(query_uids, query_idx, sort_ids):
        Avep, rr, itself = [], 0, False
        for j, index in enumerate(list(sort_id)):
            if key_uids[index] == label:
                if key_idx[index] == idx:
                    itself = True
                    continue
                else:
                    ranks = j if itself else j + 1
                    Avep.append((len(Avep) + 1) / (ranks))
                    if rr == 0:
                        rr = 1 / (ranks)
            if j > 10000:
                break
        
        if len(Avep) != 0:
            MAP.append(sum(Avep) / len(Avep))
        else:
            continue
        MRR.append(rr)
    
    result = {
        '{}_MAP'.format(prefix): round(float(np.mean(MAP)),6),
        '{}_MRR'.format(prefix): round(float(np.mean(MRR)),6),
    }
    return result

def evaluate(args: Any, model: torch.nn.Module, tokenizer: Any) -> Dict[str, float]:
    """Evaluate the model on the given query and candidate data and compute metrics.
    Args:
        args: Arguments containing configuration parameters
        model: Model to evaluate
        tokenizer: Tokenizer used for preprocessing
    Returns:
        Dictionary containing evaluation metrics like MAP and MRR
    """
    modes = args.mode.split('2')

    query_dataset = TextDataset(tokenizer, args, args.query_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    query_cache = os.path.join(args.cache_dir, f'Vector-{args.model}-{args.dataset}-{re.split(r"[/.]", args.query_file)[-2]}-{modes[0]}.pkl')
    
    code_dataset = TextDataset(tokenizer, args, args.candidate_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)
    code_cache = os.path.join(args.cache_dir, f'Vector-{args.model}-{args.dataset}-{re.split(r"[/.]", args.candidate_file)[-2]}-{modes[1]}.pkl')    
    
    # Eval!
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = time.perf_counter()
    if os.path.exists(query_cache):
        query_vecs, query_urls, query_idxs = pickle.load(open(query_cache,'rb'))
        # pass
    else:
        query_vecs, query_urls, query_idxs = do_evaluate(args, model, modes[0], query_dataloader)
        pickle.dump((query_vecs, query_urls, query_idxs), open(query_cache,'wb'))
    
    if os.path.exists(code_cache):
        key_vecs, key_urls, key_idxs = pickle.load(open(code_cache,'rb'))
        # pass
    else:
        key_vecs, key_urls, key_idxs = do_evaluate(args, model, modes[1], code_dataloader)
        pickle.dump((key_vecs, key_urls, key_idxs), open(code_cache,'wb'))
    embedding_time = time.perf_counter() - start_time

    # return None
    result = do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, args.mode, re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
                     query_vecs=query_vecs, key_vecs=key_vecs, query_uids=query_urls, key_uids=key_urls, query_idx=query_idxs, key_idx=key_idxs)
    total_time = time.perf_counter() - start_time
    logging.info(('Eval Finish, Total %.6f second, Embedding Time %.6f second', total_time, embedding_time))
    return result

def json_pretty_dump(obj: Any, filename: str) -> None:
    """Dump JSON object to file with pretty formatting.
    Args:
        obj: JSON-serializable object to dump
        filename: Path to output file
    """
    obj = vars(obj)
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False)        
                        
def main():
    """Main function for running the evaluation script.
    Parses command line arguments, sets up logging, initializes the model,
    tokenizer, and runs the evaluation process.
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--cache_dir", default='./cache', type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default='./result', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--query_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--candidate_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--cls", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset', type=str, default=None,
                        help="random seed for initialization")
    parser.add_argument('--mode', type=str, default='nlp2code', choices=['nlp2code', 'nlp2nlp', 'code2code', 'nlp+code2code+code','remix2code'],
                        help="random seed for initialization")
    parser.add_argument('--model', type=str, default='UniCoR',
                        help="random seed for initialization")
    
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    logger.info("Training/evaluation parameters %s", args)

    model = Model(args)
    tokenizer = model.tokenizer
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model) 

    model.to(args.device)
    result = evaluate(args, model, tokenizer)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],6)))
    save_json_data(args.output_dir, f"result-{args.model}-{args.dataset}-{args.mode}-{re.split(r'[/.]', args.query_file)[-2]}2{re.split(r'[/.]', args.candidate_file)[-2]}.jsonl", result)

if __name__ == "__main__":
    main()


