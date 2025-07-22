import os
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
            if type(data[0]) in [str, list,dict]:
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

def do_metric(prefix, scores, query_uids, key_uids, query_idx=None, key_idx=None):
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query1_path", default=None, type=str, 
                        help="nlp")
    parser.add_argument("--target1_path", default=None, type=str, 
                        help="code")
    parser.add_argument("--query2_path", default=None, type=str, 
                        help="nlp")
    parser.add_argument("--target2_path", default=None, type=str, 
                        help="code")
    parser.add_argument("--query3_path", default=None, type=str, 
                        help="nlp")
    parser.add_argument("--target3_path", default=None, type=str, 
                        help="code")
    parser.add_argument("--prefix1", default=None, type=str, 
                        help="code")
    parser.add_argument("--prefix2", default=None, type=str, 
                        help="code")

    parser.add_argument("--output_dir",default='./result/result-codebridge', type=str,help="embedding/eval")

    #print arguments
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'-cache', exist_ok=True)
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    
    if os.path.exists(os.path.join(args.output_dir+'-cache', f"result-{args.prefix1}.jsonl")):
        print('{} is already'.format(args.prefix1))
    else:
        print('{} is doing...'.format(args.prefix1))   
        # test different models and settings 
        query1_vecs, query1_urls, query1_idxs = pickle.load(open(args.query1_path,'rb'))
        query2_vecs, query2_urls, query2_idxs = pickle.load(open(args.query2_path,'rb'))
        query3_vecs, query3_urls, query3_idxs = pickle.load(open(args.query3_path,'rb'))

        target1_vecs, target1_urls, target1_idxs = pickle.load(open(args.target1_path,'rb'))
        target2_vecs, target2_urls, target2_idxs = pickle.load(open(args.target2_path,'rb'))
        target3_vecs, target3_urls, target3_idxs = pickle.load(open(args.target3_path,'rb'))

        query = np.concatenate([query1_vecs, query2_vecs, query3_vecs], axis=-1)
        target = np.concatenate([target1_vecs, target2_vecs, target3_vecs], axis=-1)
        score = np.matmul(query, target.T)

        res = do_metric(args.prefix1, score, query1_urls, target1_urls, query1_idxs, target1_idxs)
        best_concat = {**res, 'weight1':'-', 'weight2':'-', 'weight3':'-'}
        save_json_data(args.output_dir, f"result-{args.prefix2}.jsonl", best_concat)

        scores1=np.matmul(query1_vecs,target1_vecs.T)
        scores2=np.matmul(query2_vecs,target2_vecs.T)
        scores3=np.matmul(query3_vecs,target3_vecs.T)

        step_size = 0.05
        best_score, best = -1, None
        ans_list, weight_list = [], []
        for w1 in np.arange(0, 1+step_size,step_size):
            for w2 in np.arange(0, 1+step_size, step_size):
                w3 = 1 - w1 - w2
                if w3 < 0 :
                    continue
                weight_list.append((w1, w2, w3))

        for w1, w2, w3 in tqdm(weight_list, desc=args.prefix1):
            res = do_metric(args.prefix1, w1*scores1 + w2*scores2 + w3*scores3, query1_urls, target1_urls, query1_idxs, target1_idxs)
            map_score = res.get(f"{args.prefix1}_MAP", 0)
            if map_score > best_score:
                best_score = map_score
                best = {**res, 'weight1':w1, 'weight2':w2, 'weight3':w3}
            ans_list.append(res)
        
        save_json_data(args.output_dir+'-cache', f"result-{args.prefix1}.jsonl", ans_list)
        save_json_data(args.output_dir, f"result-{args.prefix1}.jsonl", best)
    
    
   