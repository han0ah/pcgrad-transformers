# pcgrad-transformers
Implementaion of experiment for applying [PCGrad](https://arxiv.org/abs/2001.06782) method on robeta-based model &amp; multi NLP tasks. Refer to [https://han0ahblog.tistory.com/2](https://han0ahblog.tistory.com/2) for details.

### Note
[pcgrad.py](pcgrad.py) is copy of https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

### Environment
```
python 3.10
pytorch==1.13.1
transformers==4.25.1
```
### Setting
#### Task
* [Task l : PAWS-KR](https://github.com/monologg/KoELECTRA/tree/master/finetune/data/paws)
* [Task 2 : KLUE-NLI](https://klue-benchmark.com/tasks/68/data/description)

#### Shared/Core PLM Model
* [klue/roberta-base](https://huggingface.co/klue/roberta-base)

### Run
```
python train.py # for baseline
python train_pcgrad.py # with pcgrad
```
### Performance
Validation Loss
|          | baseline | +pcgrad |
|----------|----------|---------|
| PAWS-KR  | 0.4793   | **0.4071** |
| KLUE-NLI | 0.4432   | **0.4365** |

Validation Accuracy
|          | baseline | +pcgrad |
|----------|----------|---------|
| PAWS-KR  | 0.8030   | **0.8325** |
| KLUE-NLI | 0.8486   | **0.8520** |


### Reference
* [Yu et al., Gradient Surgery for Multi-Task Learning. NeurIPS. 2020](https://arxiv.org/abs/2001.06782)
* https://github.com/WeiChengTseng/Pytorch-PCGrad

