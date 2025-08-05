# UniCoR
The data and the source code of UniCoR, Modality Collaboration for Robust Cross-Language Hybrid Code Retrieval, are available in the repository.

## Structure
```
├── dataset
├   ├── Train
│   ├── XCE
│   ├── ...(Empirical)
├── checkpoint
├── script
├── README.md
├── requirements.txt
```

## Data
UniCoR is designed for evaluating Cross-Language Hybrid Code Retrieve. It contains selected instances from CodeJamData, AtCoder, XLCoST, CodeSearchNet and XCodeEval. In this paper, the first four datasets are used as Empirical Benchmarks for phenomenon analysis, while the last dataset contains more programming languages and a larger data volume, making it more comprehensive for verifying the true performance of models.

**Data Statistics:**
|dataset|Languages|DatasetSize|Problem|CodeTokens|NLTokens|
|----|----|----|----|----|----|
|CodeJamData | 2 | 402 | 21 | 764 | 75 |
|AtCoder | 2 | 1386 | 77 | 517 | 62 |
|XLCoST | 2 | 882 | 882 | 215 | 54 |
|CodeSearchNet | 2 | 3148 | 324 | 708 | 60 |
|XCodeEval | 11 | 20148 | 6574 | 274 | 69 |

**Data Files:**
All instances in UniCoR are in `dataset`, where `dataset/XCE` contains all XCodeEval instances and other subdirectories contains all empirical instances. In each repo，data is categorised by programming language into different files.

Each instance has fields such as `Query`, `Code`, `Url`, and `Index`. `Query` and `Code` correspond to the natural language query and code implementation of the instance, respectively. Identical `Url` indicate that the code functions are consistent.

## Installation
- Linux Machine
- CUDA == 12.8
- Python == 3.13.2
- Torch == 2.7.1

For specific information on other third-party libraries, please refer to `requirements.txt`.

## Quick Start
We provide a Checkpoint based on UniXcoder to facilitate rapid model verification in `checkpoint`. If you wish to retrain the model, please follow the script below:

```bash
bash script/train.sh
```

Evaluation
```bash
   bash script/inference_empirical.sh
   bash script/inference_XCE.sh
```