import random
import os
import numpy as np
import torch
import json
import re
import tokenize
import matplotlib.pyplot as plt
import random

from typing import List, Dict, Tuple, Any, Union, Optional, TypeVar, Callable
from io import StringIO
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

class Loss_Curve:
    """Class to track and visualize training and evaluation loss curves.
    
    Attributes:
        name: List of metric names being tracked
        loss: Dictionary storing the average loss values for each metric
        temp_loss: Dictionary storing temporary loss values before averaging
    """
    def __init__(self, name_list: List[str]) -> None:
        """Initialize Loss_Curve with the list of metric names to track.
        
        Args:
            name_list: List of metric names (e.g., ['Loss', 'Accuracy', 'Patience'])
        """
        self.name = name_list
        self.loss = {item:[] for item in self.name}
        self.temp_loss = {item:[] for item in self.name}
    
    def add(self, data: Tuple[Optional[Union[torch.Tensor, int, float]], ...]) -> None:
        """Add a new set of loss values to the temporary storage.
        
        Args:
            data: Tuple of loss values corresponding to the tracked metrics
        """
        for idx, name in enumerate(self.name):
            if data[idx] is not None:
                if name == 'Patience':
                    self.temp_loss[name].append(data[idx])
                else:
                    self.temp_loss[name].append(data[idx].mean().item())
    
    def get_loss_info(self, step: int) -> str:
        """Calculate average losses and generate a summary string.
        
        Args:
            step: Current training step number
            
        Returns:
            str: Formatted string with step number and average loss values
        """
        output =  f'Step {step}'
        for name in self.name:
            if len(self.temp_loss[name]) != 0:
                avg_loss = round(sum(self.temp_loss[name])/len(self.temp_loss[name]), 6)
                output += f'\t {name} {avg_loss}'
                self.loss[name].append(avg_loss)

        self.temp_loss = {item: [] for item in self.name}
        return output
    
    def draw_graph(self, path: str) -> None:
        """Draw and save the loss curves to the specified path.
        
        Args:
            path: Directory path where the graph image will be saved
        """
        print_plot(path, self.loss)

def remove_language(text: str, words: List[str]) -> str:
    """Remove specific words from text using regular expressions.
    
    Args:
        text: Input text from which words will be removed
        words: List of words to remove from the text
        
    Returns:
        str: Text with specified words removed
    """
    # Combine words into a regex pattern
    pattern = r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b'
    # Remove words using case-insensitive replacement
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Handle multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
     

def json_pretty_dump(obj: Any, filename: str) -> None:
    """Serialize an object to a JSON file with pretty formatting.
    
    Args:
        obj: Object to be serialized to JSON
        filename: Path to the output JSON file
    """
    obj = vars(obj)
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False)
        
def print_result(data: Dict[str, float]) -> PrettyTable:
    """Generate a formatted table of model evaluation results.
    
    Args:
        data: Dictionary containing evaluation metrics (MRR and MAP scores)
        
    Returns:
        PrettyTable: Formatted table displaying the evaluation results
    """
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

def print_plot(path: str, data: Dict[str, List[float]]) -> None:
    """Generate and save plots of training/evaluation curves.
    
    Args:
        path: Directory path where the plot image will be saved
        data: Dictionary containing metric names and their corresponding values
    """
    x = range(len(data['Loss']))
    plt.figure(figsize=(20, 12))

    for _name, _value, in data.items():
        if len(_value) > 0:
            plt.plot(x, _value, label=_name, linewidth=2)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{path}/curves.png', dpi=300, bbox_inches='tight')


def classification_evaluation(scores: np.ndarray, label: np.ndarray) -> str:
    """Evaluate classification performance using various metrics.
    
    Args:
        scores: Model prediction scores (logits or probabilities)
        label: True labels
        
    Returns:
        str: Formatted string containing precision, recall, AUC, and F1 scores
    """

    # Calculate AUC score
    auc = roc_auc_score(label, scores, average='macro', multi_class='ovr').tolist()

    # Convert scores to class predictions and calculate other metrics
    scores = np.argmax(scores, axis=-1)
    ps = precision_score(label, scores, average="macro")
    rs = recall_score(label, scores, average="macro")
    f1 = f1_score(label, scores, average="macro", zero_division=1)

    return f'pr:{ps:.6f}  rc:{rs:.6f}  auc:{auc:.6f} f1: {f1:.6f}'
    
def remove_comments_and_docstrings(source: str, lang: str) -> str:
    """Remove comments and docstrings from source code based on programming language.
    
    Args:
        source: Source code string from which comments and docstrings will be removed
        lang: Programming language of the source code
        
    Returns:
        str: Source code with comments and docstrings removed
    """
    if lang in ['python']:
        """Process Python code by removing comments and docstrings using tokenization."""
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
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        """Process other programming languages using regular expressions."""
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
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(root_node: Any, 
                        index: int = 0, 
                        index_root: int = 0, 
                        height_root: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Convert an AST node tree to a list of token indices with hierarchical information.
    
    Args:
        root_node: Root node of the abstract syntax tree
        index: Starting index for the current node
        index_root: Index of the parent node
        height_root: Height of the current node in the tree
        
    Returns:
        Tuple[List[Dict[str, Any]], int]: List of dictionaries with node information and the next available index
    """
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [{"index": index,
            "node": root_node,
            "father_index": index_root,
            "is_leaf": True,
            "height": height_root}], index
    else: 
        record = [{"index": index,
            "node": root_node,
            "father_index": index_root,
            "is_leaf": False, 
            "height": height_root}]
        self_index = index
        for child in root_node.children:
            record_item, self_index = tree_to_token_index(child, self_index + 1, index, height_root + 1)
            record += record_item
        return record, self_index  

def index_to_code_token(index: Tuple[Tuple[int, int], Tuple[int, int]], 
                        code: List[str]) -> str:
    """Convert position indices to the corresponding code token string.
    
    Args:
        index: Tuple of start and end positions, each being a tuple (line, column)
        code: List of code lines
        
    Returns:
        str: Code substring corresponding to the given indices
    """
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]   
    return s
   
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across various libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: List of JSON objects loaded from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Each line contains a separate JSON object
        samples = [json.loads(line) for line in f if line.strip()]
    return samples

def cal_r1_r5_r10(ranks: List[float], prefix: str) -> Dict[str, float]:
    """Calculate Mean Reciprocal Rank (MRR) metric.
    
    Args:
        ranks: List of rank values
        prefix: Prefix for the metric name in the result dictionary
        
    Returns:
        Dict[str, float]: Dictionary containing the MRR score
    """
    result = {'{} MRR'.format(prefix): round(float(np.mean(ranks)), 6)}
    return result
    
def save_json_data(save_dir: str, filename: str, data: Union[Dict[str, Any], List[Any], List[Dict[str, Any]]]) -> None:
    """Save data to a JSON file with appropriate formatting.
    
    Args:
        save_dir: Directory path where the JSON file will be saved
        filename: Name of the output JSON file
        data: Data to be saved (typically a dictionary or list)
        
    Raises:
        RuntimeError: If the data type is not supported
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
            raise RuntimeError('Unsupported type: %s' % type(data))