''' 
_*_ coding: utf-8 _*_
Date: 2020/6/26
Author: YZL
Intent: MCQA robert model
'''

from transformers import BertModel, BertLayer
import torch.nn as nn
import torch


class FirstModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['pretrain_model_dir'])  # pretrain_model_dir2
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, sample):
        inputs_ids, inputs_masks = sample['inputs_ids'], sample['inputs_masks']
        types_ids = sample['type_ids']

        cls = self.bert(inputs_ids, attention_mask=inputs_masks, token_type_ids=types_ids)[1]
        logits = self.linear(cls)
        score = torch.sigmoid(logits).squeeze(-1)
        return score


if __name__ == "__main__":
    from lajs_config import LajsConfig
    from lajs_datasets import SegmentSelectionDataset, select_collate
    from torch.utils.data import DataLoader
    d = SegmentSelectionDataset(LajsConfig, LajsConfig['train_file'])
    loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=2, collate_fn=select_collate)
    model = FirstModel(LajsConfig)
    for i, item in enumerate(loader):
        # print(item)
        print('*' * 20)
        print(model(item))
        if i > 4:
            break

    pass