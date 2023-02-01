from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

    
class QuoteCSE_Dataset(Dataset):
    def __init__(self, args, original_texts, positive_texts, negative_texts):
        self.tokenizer = args.tokenizer
        self.org = []
        self.pos = []
        self.neg = []
        self.max_len = args.max_len
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

    def __len__(self):
        return len(self.org)

    def __getitem__(self, idx):
        return self.org[idx], self.pos[idx], self.neg[idx]
    



def create_data_loader(args, df, shuffle, drop_last):
    cd = QuoteCSE_Dataset(
        args,
        original_texts = df.title_quote.to_numpy(),
        positive_texts = df.positive_sentence.to_numpy(),
        negative_texts = df.negative_sentence.to_numpy(),
    )

    return DataLoader(
        cd,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = shuffle,
        drop_last = drop_last
    )


def tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                  batch[0]['attention_mask'].to(device, dtype=torch.long),
                  batch[1]['input_ids'].to(device, dtype=torch.long),
                  batch[1]['attention_mask'].to(device, dtype=torch.long),
                  batch[2]['input_ids'].to(device, dtype=torch.long),
                  batch[2]['attention_mask'].to(device, dtype=torch.long)])
