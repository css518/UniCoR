import json
import os
import re
import pandas as pd
import argparse
import numpy as np

def load_results(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r') as f:
                try:
                    results.append(json.loads(f.readline()))
                except:
                    continue
    return results

def load_cache(directory, weight=None):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r') as f:
                for idx, lines in enumerate(f):
                    results.append({**json.loads(lines), 'weight1':weight[idx][0], 'weight2':weight[idx][1], 'weight3':weight[idx][2]})
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='./result/result-all-Codebridge', type=str, 
                        help="nlp")
    parser.add_argument("--output_dir",default='./result/model_performance-Codebridge.csv', type=str)

    #print arguments
    args = parser.parse_args()
    data, weight_list, cache = [], [], []


    raw = load_results(args.path)
    # print('load data 1')

    step_size = 0.05
    for w1 in np.arange(0, 1+step_size,step_size):
        for w2 in np.arange(0, 1+step_size, step_size):
            w3 = 1 - w1 - w2
            if w3 < 0 :
                continue
            weight_list.append((w1, w2, w3))

    raw_fix = load_cache(args.path+'-cache', weight_list)

    for data_item in raw_fix:
        if data_item['weight1'] == 0.25 and data_item['weight2'] == 0.65:
            print(data_item)
            raw.append(data_item)
        
        weight1 = data_item['weight1']
        weight2 = data_item['weight2']
        weight3 = data_item['weight3']
        for name, value in data_item.items():
            if 'weight' in name :
                continue
            names = re.split(r'[-_]', name)

            cache.append({
                'Model': names[0],
                'Dataset': names[1],
                'Way': names[2],
                'Lang': names[3],
                'Metric': names[-1],
                'Value':  value,
                'Weight1': weight1,
                'Weight2': weight2,
                'Weight3': weight3,
            })
    
    df_cache = pd.DataFrame(cache)
    result_cache_rows = []
    for (dataset, model, way, v1, v2, v3), groups in df_cache.groupby(['Dataset', 'Model', 'Way', 'Weight1', 'Weight2', 'Weight3']):
        row = {
            'Dataset': dataset,
            'Model': model,
            'Way': way,
            'Weight1': v1,
            'Weight2': v2,
            'Weight3': v3,
        }
        for _, group in groups.iterrows():
            row['{} {}'.format(group['Lang'], group['Metric'])] = group['Value']
        result_cache_rows.append(row)

    df_cache_new = pd.DataFrame(result_cache_rows)
    
    group = ['Dataset', 'Model', 'Way']
    map_cols = [col for col in df_cache_new.columns if 'MAP' in col]
    mrr_cols = [col for col in df_cache_new.columns if 'MRR' in col]

    df_cache_new['MRR'] = df_cache_new[mrr_cols].mean(axis=1)
    df_cache_new['MAP'] = df_cache_new[map_cols].mean(axis=1)

    idx = df_cache_new.groupby(group)['MRR'].idxmax()
    best_df = df_cache_new.loc[idx, :]

    for item in raw:
        weight1 = item['weight1']
        weight2 = item['weight2']
        weight3 = item['weight3']
        for name, value in item.items():
            if 'weight' in name :
                continue
            names = re.split(r'[-_]', name)

            data.append({
                'Model': names[0],
                'Dataset': names[1],
                'Way': names[2],
                'Lang': names[3],
                'Metric': names[-1],
                'Value':  value,
                'Weight1': weight1,
                'Weight2': weight2,
                'Weight3': weight3,
            })


    df = pd.DataFrame(data)
    result_rows = []
    for (dataset, model, way, v1, v2, v3), groups in df.groupby(['Dataset', 'Model','Way' , 'Weight1', 'Weight2', 'Weight3']):
        row = {
            'Dataset': dataset,
            'Model': model,
            'Way': way,
            'Weight1': v1,
            'Weight2': v2,
            'Weight3': v3,
        }
        for _, group in groups.iterrows():
            row['{} {}'.format(group['Lang'], group['Metric'])] = group['Value']
        result_rows.append(row)

    df_new = pd.DataFrame(result_rows)
    map_cols = [col for col in df_new.columns if 'MAP' in col]
    mrr_cols = [col for col in df_new.columns if 'MRR' in col]

    df_new['MRR'] = df_new[mrr_cols].mean(axis=1)
    df_new['MAP'] = df_new[map_cols].mean(axis=1)

    df_concat = pd.concat([df_new, best_df], ignore_index=True)

    fix_cols = ['Dataset', 'Model', 'Way', 'Weight1', 'Weight2', 'Weight3', 'MRR', 'MAP']

    other_cols = [col for col in df_concat.columns if col not in fix_cols]

    df_concat = df_concat[fix_cols + sorted(other_cols)]
    df_concat.to_csv(args.output_dir, index=False)
    print(f"\nSave Data to {args.output_dir}")