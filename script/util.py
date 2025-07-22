import random
import os
import numpy as np
import torch
import json
import re
import  tokenize
import matplotlib.pyplot as plt

from typing import List, Dict
from io import StringIO
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score,  roc_auc_score, precision_score, recall_score, f1_score

class Loss_Curve:
    def __init__(self, name_list) -> None:
        self.name = name_list
        self.loss = {item:[] for item in self.name}
        self.temp_loss = {item:[] for item in self.name}
    
    def add(self, data:tuple) -> None:
        for idx, name in enumerate(self.name):
            if data[idx] is not None:
                if name == 'Patience':
                    self.temp_loss[name].append(data[idx])
                else:
                    self.temp_loss[name].append(data[idx].mean().item())
    
    def get_loss_info(self, step) -> str:
        output =  f'Step {step}'
        for name in self.name:
            if len(self.temp_loss[name]) != 0:
                avg_loss = round(sum(self.temp_loss[name])/len(self.temp_loss[name]),6)
                output += f'\t {name} {avg_loss}'
                self.loss[name].append(avg_loss)

        self.temp_loss = {item:[] for item in self.name}
        return output
    
    def draw_graph(self, path) -> None:
        print_plot(path, self.loss)

def remove_language(text, words):
    # 将多个单词组合成一个正则表达式模式
    pattern = r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b'
    # 使用re.IGNORECASE标志忽略大小写
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # 处理可能出现的多个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
     

def json_pretty_dump(obj, filename):
    obj = vars(obj)
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False)
        
def print_result(data:json) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ['Field', 'code2code_MRR', 'code2code_MAP', 'nl2code_MRR', 'nl2code_MAP']
    table.align["Field"] = "l"
    table.align["code2code_MRR"] = "r"
    table.align["nl2code_MRR"] = "r"
    table.align["code2code_MAP"] = "r"
    table.align["nl2code_MAP"] = "r"

    if 'code2code_MRR' in data:
        table.add_row(['all', data['code2code_MRR'], data['code2code_MAP'], data['nl2code_MRR'], data['nl2code_MAP']])

    lines = list(set([_item.split(' ')[0] for _item in data.keys() if _item.split(' ')[0] not in ['code2code_MRR', 'nl2code_MRR', 'code2code_MAP', 'nl2code_MAP']]))
    lines.sort()

    for name in lines:
        table.add_row([name, data[f'{name} code2code_MRR'], data[f'{name} code2code_MAP'], data[f'{name} nl2code_MRR'], data[f'{name} nl2code_MAP']])

    return table

def print_plot(path, data:json) -> None:
    x = range(len(data['Loss']))
    plt.figure(figsize=(20, 12))

    for _name, _value, in data.items():
        if len(_value) > 0:
            plt.plot(x, _value, label=_name, linewidth=2)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{path}/curves.png', dpi=300, bbox_inches='tight')


def classification_evaluation(scores:list, label:list) -> str:

    # ap = average_precision_score(label, scores, average='macro').tolist()
    auc = roc_auc_score(label, scores, average='macro', multi_class='ovr').tolist()

    scores = np.argmax(scores, axis=-1)
    ps = precision_score(label, scores, average="macro")
    rs = recall_score(label, scores, average="macro")
    effection = f1_score(label, scores, average="macro", zero_division=1)

    return f'pr:{ps:.6f}  rc:{rs:.6f}  auc:{auc:.6f} f1: {effection:.6f}'
    
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(root_node, index:int=0, index_root:int=0, height_root: int=0):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [{"index": index,
            "node": root_node,
            "father_index": index_root,
            "is_leaf": True,
            "height": height_root}], index
    else: 
        record=[{"index": index,
            "node": root_node,
            "father_index": index_root,
            "is_leaf": False, 
            "height": height_root}]
        self_index = index
        for child in root_node.children:
            record_item, self_index = tree_to_token_index(child, self_index + 1, index, height_root+1)
            record+=record_item
        return record, self_index  

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
   
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True

def load_json(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 每行是一个独立的JSON对象
        samples = [json.loads(line) for line in f if line.strip()]
    return samples

def cal_r1_r5_r10(ranks, prefix):
    result = {'{} MRR'.format(prefix):round(float(np.mean(ranks)),6)}
    return result
    
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