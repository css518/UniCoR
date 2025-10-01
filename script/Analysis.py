import json
import os
import re
import pandas as pd
import argparse
from typing import List, Dict, Any
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, 
                        help="Path to the directory containing JSONL files")
    parser.add_argument("--output_dir",default=None, type=str,
                        help="Path to save result CSV file")

    args = parser.parse_args()

    #loading results from JSONL files
    raw = load_results(args.path)
    data = []
    # ZC3-AtCoder-code2code-java2java_MAP
    for name, value in raw.items():
        flag = 'My' in name
        names = re.split(r'[-_]', name)
        if flag:
            data.append({
                'Model': 'My-' + names[1],
                'Dataset': names[2],
                'Way': names[3],
                'Lang': names[4],
                'Metric': names[-1],
                'Value':  value
            })
        else:
            data.append({
                'Model': names[0],
                'Dataset': names[1],
                'Way': names[2],
                'Lang': names[3],
                'Metric': names[-1],
                'Value':  value
            })

    #data to DataFrame & calculate mean
    df = pd.DataFrame(data)
    result_rows = []
    for (dataset, model, way), groups in df.groupby(['Dataset', 'Model', 'Way']):
        row = {
            'Dataset': dataset,
            'Model': model,
            'Way': way,
            'Weight' : '-'
        }
        for _, group in groups.iterrows():
            row['{} {}'.format(group['Lang'], group['Metric'])] = group['Value']
        result_rows.append(row)

    df_new = pd.DataFrame(result_rows)

    map_cols = [col for col in df_new.columns if 'MAP' in col]
    mrr_cols = [col for col in df_new.columns if 'MRR' in col]

    df_new['MRR'] = df_new[mrr_cols].mean(axis=1)
    df_new['MAP'] = df_new[map_cols].mean(axis=1)


    #save to CSV file
    fix_cols = ['Dataset', 'Model', 'Way', 'Weight', 'MRR', 'MAP']

    other_cols = [col for col in df_new.columns if col not in fix_cols]

    df_new = df_new[fix_cols + sorted(other_cols)]

    df_new.to_csv(args.output_dir, index=False)
    print(f"\nSave Data to {args.output_dir}")
    