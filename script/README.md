# Scripts

## File Descriptions
- `eval***.py`: Script for evaluating different model performance on a dataset Under (NL2Code/Code2Code/Remix/Concat) Scenarios.
- `search_best.py`: Script for searching the best weight Under Weight Scenarios.
- `Analysis***.py`: Script for analyzing the results of the evaluation script.
- `train.py`: Script for training a UniCoR model on a dataset.
- `utils.py`: Utility functions used by other scripts.
- `model.py`: Model definition and related functions.

## Usage

### Training
To train a UniCoR model on a dataset, run the `train.py` script with the appropriate command-line arguments, just like `train.sh`:
```
bash script/train.sh
```

### Evaluation
To evaluate the performance of a trained model on a dataset, run the `eval***.py` and `search_best.py` script with the appropriate command-line arguments, just like `inference.sh`:
```
bash script/inference_empirical.sh  # for empirical benchmarks 
bash script/inference_XCodeEval.sh  # for XCodeEval benchmarks
bash script/inference_rebuttal.sh  # add experiment about BM25 and text-embedding of OpenAI during rebuttal
```

Notice:  if you want to evaluate the performance of commercial models, please fill your OpenAI API key in main function of `eval_openai.py`
