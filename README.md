# UniCoR
Here are the Pytorch implementation and data of UniCoR in the ICSE 2026: *Modality Collaboration for Robust Cross-Language Hybrid Code Retrieval*. 

## Dependencies

- Ubuntu 18.04
- CUDA == 12.8
- Python == 3.13.2
- Torch == 2.7.1

For specific information on other third-party libraries, please refer to `requirements.txt`. please run the following command to install them:

```bash
conda create -n unicor python=3.13.2
conda activate unicor
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Structure
```
├── dataset
├   ├── Train
│   ├── XCodeEval
│   ├── ...(Empirical)
├── checkpoint
├── script
├── README.md
├── requirements.txt
```

### Dataset
UniCoR is designed for evaluating Cross-Language Hybrid Code Retrieve. It contains selected instances from CodeJamData, AtCoder, XLCoST, CodeSearchNet and XCodeEval. In this paper, the first four datasets are used as Empirical Benchmarks for phenomenon analysis, while the last dataset(XCodeEval) contains more programming languages and a larger data volume, making it more comprehensive for verifying the true performance of models.

**evaluation data statistics:**
|dataset|Languages|DatasetSize|Problem|CodeTokens|NLTokens|
|----|----|----|----|----|----|
|CodeJamData | 2 | 402 | 21 | 764 | 75 |
|AtCoder | 2 | 1386 | 77 | 517 | 62 |
|XLCoST | 2 | 882 | 882 | 215 | 54 |
|CodeSearchNet | 2 | 3148 | 324 | 708 | 60 |
|XCodeEval | 11 | 20148 | 6574 | 274 | 69 |

**evaluation data files:**
All instances in UniCoR are in `dataset`, where `dataset/XCE` contains all XCodeEval instances and other subdirectories contains all empirical instances. In each repo，data is categorised by programming language into different files.

Each instance has fields such as `Query`, `Code`, `Url`, and `Index`. `Query` and `Code` correspond to the natural language query and code implementation of the instance, respectively. Identical `Url` indicate that the code functions are consistent.

### Parser

This folder comes from the [GraphCodeBert](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT), which is used to parse data flow graphs.

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```bash
cd parser
bash build.sh
cd ..
```

### Checkpoint
We provide a Checkpoint based on UniXcoder to facilitate rapid model verification in `checkpoint`. 
This includes weight files, word segmentation models, and parameter files



## Quick Start
If you wish to retrain the model, please follow the script below:

```bash
   bash script/train.sh
```

Evaluation
```bash
   bash script/inference_empirical.sh
   bash script/inference_XCodeEval.sh
```

You may refer to  [Script Readme](./script/README.md) for more details.

## Citation

If you find this repo or our work helpful, please consider citing us:
```bibtex
@inproceedings{yang2026unicor,
   title={UniCoR: Modality Collaboration for Robust Cross-Language Hybrid Code Retrieval},
   author={Yang Yang and Li Kuang and Jiakun Liu and Zhongxin Liu and Yingjie Xia and David Lo},
   booktitle = {Proceedings of the 48th International Conference on Software Engineering},
   year={2026},
   url = {https://doi.org/10.1145/3744916.3773201},
   doi = {10.1145/3744916.3773201},
   location = {Rio de Janeiro, Brazil},
   series = {ICSE '26}
}
```



