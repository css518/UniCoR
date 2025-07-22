"""
clean version of evaluation
"""
import os
import torch
from tqdm import tqdm
import logging
import numpy as np
import argparse
import pickle
import json

def save_json_data(save_dir, filename, data):
    """
    将数据保存为JSON文件
    
    Args:
        save_dir (str): 保存目录路径
        filename (str): 文件名
        data: 要保存的数据(通常是字典或列表)
    """
    os.makedirs(save_dir, exist_ok=True)
        
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        if type(data) == list:
            if type(data[0]) in [str, list, dict]:
                for item in data:
                    f.write(json.dumps(item))
                    f.write('\n')

            else:
                json.dump(data, f)
        elif type(data) == dict:
            json.dump(data, f)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    print("saved dataset in " + filename)

def do_metric(prefix, scores, query_uids, key_uids, query_idx=None, key_idx=None, w1=None, w2=None):
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
        'NL2Code_weight': w1,
        'Code2Code_weight': w2,
    }
    return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query1_path", default=None, type=str, 
                        help="nlp")
    parser.add_argument("--query2_path", default=None, type=str, 
                        help="code")
    parser.add_argument("--target_path", default=None, type=str, 
                        help="code")
    parser.add_argument("--prefix", default=None, type=str, 
                        help="code")

    parser.add_argument("--output_dir",default='./result/result-weight', type=str,help="embedding/eval")

    #print arguments
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'-cache', exist_ok=True)
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    
    if os.path.exists(os.path.join(args.output_dir+'-cache', f"result-{args.prefix}.jsonl")):
        print('{} is already'.format(args.prefix))
    else:    
        # test different models and settings 
        print('{} is dealing'.format(args.prefix))
        query1_vecs, query1_urls, query1_idxs = pickle.load(open(args.query1_path,'rb'))
        query2_vecs, query2_urls, query2_idxs = pickle.load(open(args.query2_path,'rb'))

        key_vecs, key_urls, key_idxs = pickle.load(open(args.target_path,'rb'))

        scores1=np.matmul(query1_vecs,key_vecs.T)
        scores2=np.matmul(query2_vecs,key_vecs.T)

        step_size = 0.05
        best_score_MRR, best_MRR = -1, None
        best_score_MAP, best_MAP = -1, None
        # traverse all the weights, where w1+w2+w3=1, with step_size
        ans_list = []
        for w1 in tqdm(np.arange(0, 1+step_size,step_size), desc=args.prefix):
            w2 = 1 - w1
            if w2 < 0 :
                continue
            res = do_metric(args.prefix, w1*scores1 + w2*scores2, query2_urls, key_urls, query2_idxs, key_idxs, w1, w2)
            map_score = res.get(f"{args.prefix}_MAP", 0)
            mrr_score = res.get(f"{args.prefix}_MRR", 0)
            if map_score > best_score_MAP:
                best_score_MAP = map_score
                best_MAP = res
            if mrr_score > best_score_MRR:
                best_score_MRR = mrr_score
                best_MRR = res
            ans_list.append(res)
        
        save_json_data(args.output_dir+'-cache', f"result-{args.prefix}.jsonl", ans_list)
        save_json_data(args.output_dir, f"result-{args.prefix}_best_MRR.jsonl", best_MRR)
        save_json_data(args.output_dir, f"result-{args.prefix}_best_MAP.jsonl", best_MAP)
    
    
   