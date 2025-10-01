import json
import os
import re
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple, Optional

def load_results(directory: str) -> List[Dict[str, Any]]:
    """Load JSONL result files from a directory.
    Args:
        directory: Path to the directory containing JSONL files
    Returns:
        List containing all loaded results
    """
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r') as f:
                try:
                    # Read the first line of each JSONL file and parse it as a JSON object
                    results.append(json.loads(f.readline()))
                except:
                    # Skip files that fail to parse
                    continue
    return results

def load_cache(directory: str, weight: Optional[List[Tuple[float, float, float]]] = None) -> List[Dict[str, Any]]:
    """Load result files from cache directory and add weight information.
    Args:
        directory: Path to the directory containing cache JSONL files
        weight: List of weights, each element is a tuple containing three weight values
    Returns:
        List containing all loaded results with weight information
    """
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r') as f:
                for idx, lines in enumerate(f):
                    if lines  == '':
                        continue
                    else:
                        results.append(json.loads(lines))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, 
                        help="Path to the directory containing JSONL files")
    parser.add_argument("--output_dir",default=None, type=str,
                        help="Path to save result CSV file")

    args = parser.parse_args()
    data, weight_list, cache = [], [], []
    raw = load_results(args.path)
    step_size = 0.05
    for w1 in np.arange(0, 1+step_size,step_size):
        for w2 in np.arange(0, 1+step_size, step_size):
            w3 = 1 - w1 - w2
            if w3 < 0 :
                continue
            weight_list.append((w1, w2, w3))

    raw_fix = load_cache(args.path+'-cache', weight_list)

    for data_item in raw_fix:
        weight = data_item['NL2Code_weight'] if 'NL2Code_weight' in data_item else data_item['weight']
        for name, value in data_item.items():
            if 'weight' in name :
                continue
            names = re.split(r'[-_]', name)

            if 'My' in name:
                cache.append({
                    'Model': 'My-' + names[1],
                    'Dataset': names[2],
                    'Way': names[3],
                    'Lang': names[4],
                    'Metric': names[-1],
                    'Value':  value,
                    'Weight': weight
                })
            else:
                cache.append({
                    'Model': names[0],
                    'Dataset': names[1],
                    'Way': names[2],
                    'Lang': names[3],
                    'Metric': names[-1],
                    'Value':  value,
                    'Weight': weight
                })
    

    df_cache = pd.DataFrame(cache)
    result_cache_rows = []
    for (dataset, model, way, v1), groups in df_cache.groupby(['Dataset', 'Model', 'Way', 'Weight']):
        row = {
            'Dataset': dataset,
            'Model': model,
            'Way': way,
            'Weight': v1
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

    fix_cols = ['Dataset', 'Model', 'Way', 'Weight', 'MRR', 'MAP']
    other_cols = [col for col in best_df.columns if col not in fix_cols]

    df_concat = best_df[fix_cols + sorted(other_cols)]

    df_concat.to_csv(args.output_dir, index=False)
    print(f"\n结果已保存到 {args.output_dir}")
    