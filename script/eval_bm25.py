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
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for an example.
    
    Attributes:
        code: Code String
        code_tokens: Tokenized code
        nl_tokens: Tokenized natural language
        tokens: concatenated tokenized code and tokenized natural language
        url: URL or identifier for the example
    """
    def __init__(self,
                 code: str,
                 code_tokens: list[str],
                 nl_tokens: list[str],
                 tokens: list[str],
                 url: str,
    ):
        self.code = code
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.tokens = tokens
        self.url = url

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

        
def convert_examples_to_features(js):
    """convert examples to token ids"""

    code_tokens =  js['func'].split() if "func" in js else js['code'].split()
    nl_tokens = js['query'] if type(js['query']) is list else js['query'].split()
    code = ' '.join(code_tokens)

    tokens = nl_tokens + code_tokens

    return InputFeatures(code, code_tokens, nl_tokens, tokens, js['label'] if "label" in js else js["url"].split('-')[0])

class TextDataset(Dataset):
    """A single training/test dataset."""
    def __init__(self, args: Any, file_path: Optional[str] = None):
        """Initialize the TextDataset with data from the specified file.
        Args:
            args: Arguments containing configuration parameters
            file_path: Path to the data file
        """
        self.examples = []
        self.code, self.nl, self.url, self.raw = [], [], [], []
        self.dataset = args.dataset
        self.name = 'param' if 'My-Param' in args.model else args.model.lower()
        self.lang = re.split(r'[/.]', file_path)[-2]

        cache_file = os.path.join(args.cache_dir, f'dataset-{self.name}-{args.dataset}-{self.lang}.pkl')

        if os.path.exists(cache_file):
            logger.info(f' Loading dataset from {cache_file} ... ')
            (self.examples, self.nl, self.code,self.url, self.raw)=pickle.load(open(cache_file,'rb'))
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
                _item = convert_examples_to_features(js)
                self.examples.append(_item)
                self.nl.append(_item.nl_tokens)
                self.code.append(_item.code_tokens)
                self.url.append(_item.url) 
                self.raw.append(_item.code)
            
            pickle.dump((self.examples, self.nl, self.code,self.url, self.raw),open(cache_file,'wb'))
                                        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, _idx):
        return (self.examples[_idx].code, self.examples[_idx].code_tokens, self.examples[_idx].nl_tokens, self.examples[_idx].tokens, self.examples[_idx].url)
            

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
def do_metric(prefix: str, scores: np.ndarray, query_uids: np.ndarray, key_uids: np.ndarray, query_idx: Optional[np.ndarray] = None, key_idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute evaluation metrics like MAP (Mean Average Precision) and MRR (Mean Reciprocal Rank).
    Args:
        prefix: Prefix for metric names in the result dictionary
        scores: Scores for the query examples
        query_uids: Unique identifiers for the query examples
        key_uids: Unique identifiers for the candidate examples
        query_idx: Optional indices for the query examples
        key_idx: Optional indices for the candidate examples
    Returns:
        Dictionary containing MAP and MRR scores with the specified prefix
    """
    scores=np.array(scores)
    # print(scores.shape)
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


def evaluate(args: Any) -> Dict[str, float]:
    query_dataset = TextDataset(args, args.query_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=1,num_workers=4)
    
    code_dataset = TextDataset(args, args.candidate_file)

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))

    bm25_code = BM25Okapi(code_dataset.code)
    bm25_nl = BM25Okapi(code_dataset.nl)
    key_urls = np.array(code_dataset.url)
    key_idxs = np.array(code_dataset.raw)
    query_urls = np.array(query_dataset.url)
    query_idxs = np.array(query_dataset.raw)

    scores_nl2code, scores_code2code, scores_nl2nl, scores_remix2code = [], [], [], []
    for _item in tqdm(query_dataloader):
        scores_code2code.append(bm25_code.get_scores([tok for tup in _item[1] for tok in tup]))  #code2code
        scores_nl2code.append(bm25_code.get_scores([tok for tup in _item[2] for tok in tup]))
        scores_remix2code.append(bm25_code.get_scores([tok for tup in _item[3] for tok in tup]))
        scores_nl2nl.append(bm25_nl.get_scores([tok for tup in _item[2] for tok in tup]))

    result_code2code = do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, 'code2code', re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
                     scores=scores_code2code, query_uids=query_urls, key_uids=key_urls, query_idx=query_idxs, key_idx=key_idxs)
    result_nl2code = do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, 'nl2code', re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
                    scores=scores_nl2code, query_uids=query_urls, key_uids=key_urls, query_idx=query_idxs, key_idx=key_idxs)
    result_remix2code = do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, 'remix2code', re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
                    scores=scores_remix2code, query_uids=query_urls, key_uids=key_urls, query_idx=query_idxs, key_idx=key_idxs)
    result_nl2nl = do_metric('{}-{}-{}-{}2{}'.format(args.model, args.dataset, 'nl2nl', re.split(r'[/.]', args.query_file)[-2], re.split(r'[/.]', args.candidate_file)[-2]), \
                    scores=scores_nl2nl, query_uids=query_urls, key_uids=key_urls, query_idx=query_idxs, key_idx=key_idxs)

    result_remix2code.update(result_code2code)
    result_remix2code.update(result_nl2code)
    result_remix2code.update(result_nl2nl)
    # return None
    return result_remix2code

def json_pretty_dump(obj, filename):
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
    
    #print arguments
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    # Set seed
    set_seed(args.seed)
    args.model = 'bm25'

    #build model
    logger.info("Training/evaluation parameters %s", args)

    result = evaluate(args) 
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],6)))
    save_json_data(args.output_dir, f"result-{args.model}-{args.dataset}-{args.mode}-{re.split(r'[/.]', args.query_file)[-2]}2{re.split(r'[/.]', args.candidate_file)[-2]}.jsonl", result)

if __name__ == "__main__":
    main()


