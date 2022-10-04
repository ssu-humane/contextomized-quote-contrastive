from kobert_transformers import get_tokenizer

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


class ContrastiveDataset(Dataset):
    def __init__(self, original_texts, positive_texts, negative_texts, label, max_len=512):
        self.tokenizer = get_tokenizer()
        self.org = []
        self.pos = []
        self.neg = []
        self.label = []
        self.max_len = max_len
        assert len(original_texts) == len(positive_texts)

        for idx in tqdm(range(len(positive_texts))):
            org = original_texts[idx]
            pos = positive_texts[idx]
            neg = negative_texts[idx]

            org_input = self.tokenizer(org, padding='max_length', truncation=True,
                                       max_length=self.max_len, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])

            pos_input = self.tokenizer(pos, padding='max_length', truncation=True,
                                       max_length=self.max_len, return_tensors='pt')
            pos_input['input_ids'] = torch.squeeze(pos_input['input_ids'])
            pos_input['attention_mask'] = torch.squeeze(pos_input['attention_mask'])

            neg_input = self.tokenizer(neg, padding='max_length', truncation=True,
                                       max_length=self.max_len, return_tensors='pt')
            neg_input['input_ids'] = torch.squeeze(neg_input['input_ids'])
            neg_input['attention_mask'] = torch.squeeze(neg_input['attention_mask'])

            self.org.append(org_input)
            self.pos.append(pos_input)
            self.neg.append(neg_input)
            self.label.append(label[idx])

    def __len__(self):
        return len(self.org)

    def __getitem__(self, idx):
        return self.org[idx], self.pos[idx], self.neg[idx], torch.tensor(self.label[idx], dtype=torch.long)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_data_loader(df, MAX_LEN, batch_size, num_workers, shuffle, drop_last):
    cd = ContrastiveDataset(
        original_texts=df.title_quote.to_numpy(),
        positive_texts=df.positive_sentence.to_numpy(),
        negative_texts=df.negative_sentence.to_numpy(),
        label=df.ve.to_numpy(),
        max_len=MAX_LEN
    )

    return DataLoader(
        cd,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )


def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                  batch[0]['attention_mask'].to(device, dtype=torch.long),
                  batch[1]['input_ids'].to(device, dtype=torch.long),
                  batch[1]['attention_mask'].to(device, dtype=torch.long),
                  batch[2]['input_ids'].to(device, dtype=torch.long),
                  batch[2]['attention_mask'].to(device, dtype=torch.long)])
