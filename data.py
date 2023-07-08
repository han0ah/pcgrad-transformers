import torch
import json
import pandas as pd 
from transformers import AutoTokenizer

class KorTextPairDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_n=-1):
        self.tokenizer=tokenizer
        self.max_n = max_n
        self.datasets = self._load_data(path)
        
    def _tensorize_text_pair(self, src_texts, tgt_texts, max_length=128):
        ret = self.tokenizer(
            src_texts,
            tgt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        return ret['input_ids'], ret['attention_mask']

    def __len__(self):
        return self.datasets['labels'].shape[0]
    
    def __getitem__(self,idx):
        return {key:val[idx].clone().detach() for key,val in self.datasets.items()}

class PAWSDataset(KorTextPairDataset):
    def _load_data(self, path):
        df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
        df.dropna(inplace=True)

        if self.max_n > 0:
            df = df.head(self.max_n)

        input_ids, attention_mask = self._tensorize_text_pair(
            list(df['sentence1']), list(df['sentence2'])
        )        
        labels = torch.tensor(df['label'], dtype=torch.int64)
        return {'input_ids':input_ids, 
                'attention_mask':attention_mask, 
                'labels':labels}
    
class KlueNLIDataset(KorTextPairDataset):
    def _load_data(self, path):
        label_map = {
            'contradiction':0,
            'neutral':1,
            'entailment':2
        }

        src_txts, tgt_txts, labels = [],[],[]
        with open(path, 'r', encoding='utf-8') as f:
            dlist = json.load(f)
            if self.max_n > 0: dlist = dlist[:self.max_n]
            for d in dlist:
                src_txts.append(d['premise'])
                tgt_txts.append(d['hypothesis'])
                labels.append(label_map[d['gold_label']])

        input_ids, attention_mask = self._tensorize_text_pair(
            src_txts, tgt_txts
        )        
        labels = torch.tensor(labels)
        return {'input_ids':input_ids, 
                'attention_mask':attention_mask, 
                'labels':labels}