from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import random
import numpy as np
import torch.nn.functional as F
from sentence_transformers import util



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
        

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
  
  
  
def search_pos_neg(outs, batch_size, body_len, do_normalize=True):
    # To find quote in body text that is the most closely the title quote
    if do_normalize:
        outs = F.normalize(outs, dim=1)
    
    title_embedding = outs[:batch_size]
    sentences_embeddings = outs[batch_size:]
    positive = []
    negative = []
    
    for b in range(len(body_len)):
        i = body_len[b]
        t_embedding = title_embedding[b]
        s_embedding = sentences_embeddings[:i]
        sentences_embeddings = sentences_embeddings[i:]    
        
        cos_scores = util.pytorch_cos_sim(t_embedding, s_embedding)[0]
        cos_scores = cos_scores.cpu().detach().numpy()

        top_results = np.argpartition(-cos_scores, range(1))[0:1]
        positive.append(top_results[0])
        
        idx = [j for j in range(len(s_embedding))]
        del idx[top_results[0]]
        negative_idx = random.choice(idx)
        negative.append(negative_idx)

    return positive, negative  
  
  
  
  
  
def make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx=None, neg_idx=None):
  with torch.no_grad():
    outs = encoder(
      input_ids = torch.cat([title_id, body_ids]), 
      attention_mask = torch.cat([title_at, body_atts]),
    )

  if args.assignment == 'dynamic':
    pos_idx, neg_idx = search_pos_neg(outs, args.batch_size, body_len)

  
  pos_body_ids = []
  neg_body_ids = []
  pos_body_atts = []
  neg_body_atts = []
      
  for b in range(len(body_len)):
    i = body_len[b]
    b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
    pos_body_ids.append(b_id[pos_idx[b]])
    neg_body_ids.append(b_id[neg_idx[b]])
    pos_body_atts.append(b_at[pos_idx[b]])
    neg_body_atts.append(b_at[neg_idx[b]])
        
  pos_body_ids = torch.stack(pos_body_ids, dim=0)
  neg_body_ids = torch.stack(neg_body_ids, dim=0)
  pos_body_atts = torch.stack(pos_body_atts, dim=0)
  neg_body_atts = torch.stack(neg_body_atts, dim=0)
        
  return pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts
