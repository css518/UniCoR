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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
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
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import threading
import time
import openai
import json

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Union

def ask(text: str, model: Any, count:int = 1):
    try:
        if len(text) > 8192:
            text = text[:8192]

        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        logger.info(e)
        logger.info('NO.{} for embedding text....'.format(count))
        time.sleep(3)
        return ask(text, model, count+1)

class InputFeatures(object):
    """A single training/test features for an example.
    
    Attributes:
        code: Code string
        nl: Natural language
        remix: concat of code and nl
        url: URL or identifier for the example
    """
    def __init__(self,
                 code: str,
                 nl: str,
                 remix: str,
                 url: str,

    ):
        self.code = code
        self.nl = nl
        self.remix = remix
        self.url = url


def save_json_data(save_dir, filename, data):
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

        
def convert_examples_to_features(js):
    """Convert examples to input item for the model.
    Args:
        js: A dictionary containing the example data.
    Returns:
        InputFeatures: A single training/test features for an example.
    """
    code =  ' '.join(js['func'].split()) if "func" in js else ' '.join(js['code'].split())
    nl = ' '.join(js['query']) if isinstance(js['query'], list) else ' '.join(js['query'].split())
    remix = nl + code

    return InputFeatures(code, nl, remix, js['label'] if "label" in js else js["url"].split('-')[0])

class TextDataset(Dataset):
    """Dataset class for text data containing code and natural language examples.
    """
    def __init__(self, args: Any, file_path: Optional[str] = None):
        self.examples = []
        self.dataset = args.dataset
        self.name = 'param' if 'My-Param' in args.model else args.model.lower()
        self.lang = re.split(r'[/.]', file_path)[-2]
        self.code_length = args.code_length
        self.data_flow_length = args.data_flow_length

        cache_file = os.path.join(args.cache_dir, f'dataset-{self.name}-{args.dataset}-{self.lang}.pkl')

        if os.path.exists(cache_file):
            logger.info(f' Loading dataset from {cache_file} ... ')
            self.examples=pickle.load(open(cache_file,'rb'))
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
                elif "codebase"in file_path or "code_idx_map" in file_path:
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
                self.examples.append(convert_examples_to_features(js))
            
            pickle.dump(self.examples,open(cache_file,'wb'))                        
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, _idx):
        return (self.examples[_idx].code,self.examples[_idx].nl, self.examples[_idx].remix, self.examples[_idx].url)
            
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

def fetch_info(text: str, model: Any, result: List[str], index: int) -> None:
    """Fetch model response for a single input text and store it in the result list.
    
    Args:
        text: Input text to query the model
        model: The model instance to use for generating responses
        result: List to store model responses
        index: Index in the result list to store the response
    """
    result[index] = ask(text, model)

def accelerate_ask(inputs: List[str], model: Any) -> List[str]:
    """Accelerate the process of asking the model for responses to a list of inputs.
    
    Args:
        inputs: List of input strings to query the model
        model: The model instance to use for generating responses
    Returns:
        List[str]: List of model responses corresponding to each input
    """
    results = [None] * len(inputs)  # storing results
    # create threads
    threads = []
    for i in range(len(inputs)):
        thread = threading.Thread(target=fetch_info, args=(inputs[i], model, results, i))
        threads.append(thread)
        thread.start()  # start the thread
    # wait for all threads to complete
    for thread in threads:
        thread.join()

    return results

def do_evaluate(args: Any, model: Any, mode: str, dataloader: DataLoader):
    """Evaluate the model on a given dataset and return embeddings, urls, and indices.
    
    Args:
        args: Arguments containing configuration parameters
        model: The model instance to use for generating embeddings
        mode: Evaluation mode ('nlp', 'code', 'remix', 'code+code', 'nlp+code')
        dataloader: DataLoader providing batches of input data
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Embeddings array of shape (num_samples, embedding_dim)
            - URLs array of shape (num_samples,)
            - Indices array of shape (num_samples,)
    """
    vecs, urls, idxs = [], [], []

    for batch in tqdm(dataloader): 
        code_inputs = batch[0]
        nl_inputs = batch[1]
        remix_inputs = batch[2]

        if mode == 'nlp':
            nlvec = accelerate_ask(nl_inputs, model)
            vec = np.array(nlvec)
        elif mode == 'code':
            codevec = accelerate_ask(code_inputs, model)
            vec = np.array(codevec)
        elif mode == 'remix':
            vec = np.array(accelerate_ask(remix_inputs, model))
        elif mode == 'code+code':
            codevec = accelerate_ask(code_inputs, model)
            vec = np.concatenate([np.array(codevec), np.array(codevec)], 1)
        elif mode == 'nlp+code':
            nlvec = accelerate_ask(nl_inputs, model)
            codevec = accelerate_ask(code_inputs, model)
            vec = np.concatenate([np.array(nlvec), np.array(codevec)], 1)

        vecs.append(vec)
        urls.append(list(batch[-1]))
        idxs.append(list(batch[0]))
    
    vecs = np.concatenate(vecs, 0)
    urls = np.concatenate(urls, 0)
    idxs = np.concatenate(idxs, 0)
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

def evaluate(args):
    """Evaluate the model on the given query and candidate data and compute metrics.
    Args:
        args: Arguments containing configuration parameters
    Returns:
        Dictionary containing evaluation metrics like MAP and MRR
    """
    modes = args.mode.split('2')
    query_dataset = TextDataset(args, args.query_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)
    query_cache = os.path.join(args.cache_dir, f'Vector-{args.model}-{args.dataset}-{re.split(r"[/.]", args.query_file)[-2]}-{modes[0]}.pkl')
    
    code_dataset = TextDataset(args, args.candidate_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)
    code_cache = os.path.join(args.cache_dir, f'Vector-{args.model}-{args.dataset}-{re.split(r"[/.]", args.candidate_file)[-2]}-{modes[1]}.pkl')    

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = time.perf_counter()
    if os.path.exists(query_cache):
        query_vecs, query_urls, query_idxs = pickle.load(open(query_cache,'rb'))
    else:
        query_vecs, query_urls, query_idxs = do_evaluate(args, args.model, modes[0], query_dataloader)
        pickle.dump((query_vecs, query_urls, query_idxs), open(query_cache,'wb'))
    
    if os.path.exists(code_cache):
        key_vecs, key_urls, key_idxs = pickle.load(open(code_cache,'rb'))
    else:
        key_vecs, key_urls, key_idxs = do_evaluate(args, args.model, modes[1], code_dataloader)
        pickle.dump((key_vecs, key_urls, key_idxs), open(code_cache,'wb'))
    
    embedding_time = time.perf_counter() - start_time

    result =  do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, args.mode, re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
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
    parser.add_argument('--mode', type=str, default='nlp2code', choices=['code2nlp', 'nlp2code', 'nlp2nlp', 'code2code', 'nlp+code2code+code','remix2remix','remix2code','code+code2code','nlp+code2code'],
                        help="random seed for initialization")
    parser.add_argument('--model', type=str, default='CoCoSoDa',
                        help="random seed for initialization")
    
    args = parser.parse_args()
    

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if 'cls' in args.model:
        args.cls = True

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    # Set seed
    set_seed(args.seed)

    #build model
    logger.info("Training/evaluation parameters %s", args)
    openai.api_key = "YOUR_OPENAI_API_KEY"  # replace with your own API key

    result = evaluate(args) 
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],6)))
    save_json_data(args.output_dir, f"result-{args.model}-{args.dataset}-{args.mode}-{re.split(r'[/.]', args.query_file)[-2]}2{re.split(r'[/.]', args.candidate_file)[-2]}.jsonl", result)

if __name__ == "__main__":
    main()


