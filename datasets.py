from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
  
class QuoteCSE_Dataset(Dataset):
    def __init__(self, args, title_texts, body_texts, pos_idx, neg_idx, max_seq):
        self.tokenizer = args.tokenizer
        self.title = []
        self.body = []
        self.pos_idx = []
        self.neg_idx = []
        self.max_seq = max_seq
        self.max_len = args.max_len
        self.body_len = []
        assert len(title_texts) == len(body_texts) 
        
        for idx in tqdm(range(len(title_texts))):
            title = title_texts[idx]
            body = body_texts[idx]
            
            title_input = self.tokenizer(title, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
            title_input['input_ids'] = torch.squeeze(title_input['input_ids'])
            title_input['attention_mask'] = torch.squeeze(title_input['attention_mask'])
            
            
            body_input = self.tokenizer(body, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
            self.body_len.append(len(body_input['input_ids']))
            
            b_input, b_att, b_token = np.zeros((self.max_seq,self.max_len)), np.zeros((self.max_seq,self.max_len)), np.zeros((self.max_seq,self.max_len))
            
            b_input[:len(body_input['input_ids'])] = body_input['input_ids']
            b_att[:len(body_input['attention_mask'])] = body_input['attention_mask']
            b_token[:len(body_input['token_type_ids'])] = body_input['token_type_ids']
            
            b_input = torch.Tensor(b_input)
            b_att = torch.Tensor(b_att)
            b_token = torch.Tensor(b_token)
            
            body_input['input_ids'] = torch.squeeze(b_input)
            body_input['attention_mask'] = torch.squeeze(b_att)
            body_input['token_type_ids'] = torch.squeeze(b_token)
            
            self.title.append(title_input)
            self.body.append(body_input)
            self.pos_idx.append(pos_idx[idx])
            self.neg_idx.append(neg_idx[idx])
            
    def __len__(self):
        return len(self.title)
    
    def __getitem__(self, idx):
        return self.title[idx], self.body[idx], self.body_len[idx], torch.tensor(self.pos_idx[idx], dtype=torch.long), torch.tensor(self.neg_idx[idx], dtype=torch.long)
    



def create_data_loader(args, df, shuffle, drop_last):
    cd = QuoteCSE_Dataset(
        args,
        title_texts=df.title_quote.to_numpy(),
        body_texts=df.sentence_quotes.to_numpy(),
        pos_idx=df.pos_idx.to_numpy(),
        neg_idx=df.neg_idx.to_numpy(),
        max_seq = max(df.sentence_quotes.apply(len).values),
    )
  
    return DataLoader(
        cd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )



def tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                  batch[0]['attention_mask'].to(device, dtype=torch.long),
                  batch[1]['input_ids'].to(device, dtype=torch.long),
                  batch[1]['attention_mask'].to(device, dtype=torch.long),
                  batch[2]['input_ids'].to(device, dtype=torch.long),
                  batch[2]['attention_mask'].to(device, dtype=torch.long)])

  
