# CoCoSoDa: Effective Contrastive Learning for Code Search

Our approach adopts the pre-trained model as the base code/query encoder and optimizes it using multimodal contrastive learning and soft data augmentation. 

CoCoSoDa is comprised of the following four components:
* **Pre-trained code/query encoder** captures the semantic information of a code snippet or a natural language query and maps it into a high-dimensional embedding space. 
as the code/query encoder.
* **Momentum code/query encoder** encodes the samples (code snippets or queries) of current and previous mini-batches to enrich the negative samples.

* **Soft data augmentation** is to dynamically mask or replace some tokens in a sample (code/query) to generate a similar sample as a form of data augmentation.

* **Multimodal contrastive learning loss function** is used as the optimization objective and consists of inter-modal and intra-modal contrastive learning loss. They are used to minimize the distance of the representations of similar samples and maximize the distance of different samples in the embedding space.



## Usage

```
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
model = RobertaModel.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
```


## Reference

Shi, E., Wang, Y., Gu, W., Du, L., Zhang, H., Han, S., ... & Sun, H. (2022). [CoCoSoDa: Effective Contrastive Learning for Code Search](https://arxiv.org/abs/2204.03293). ICSE2023.
