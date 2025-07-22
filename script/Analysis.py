import json
import os
import re
import pandas as pd
import argparse

def load_results(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r') as f:
                try:
                    data = json.loads(f.readline())
                    results.update(data)
                except Exception as e:
                    print(filename, e)


    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, 
                        help="nlp")
    parser.add_argument("--output_dir",default='model_performance-02.csv', type=str)

    #print arguments
    args = parser.parse_args()

    raw = load_results(args.path)
    data = []
    # ZC3-AtCoder-code2code-java2java_MAP"
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

    fix_cols = ['Dataset', 'Model', 'Way', 'Weight', 'MRR', 'MAP']

    other_cols = [col for col in df_new.columns if col not in fix_cols]

    df_new = df_new[fix_cols + sorted(other_cols)]

    df_new.to_csv(args.output_dir, index=False)
    print(f"\nSave Data to {args.output_dir}")
    