import torch
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
        
        
        
def most_sim(outs, batch_size, body_len, do_normalize=True):
    # To find quote in body text that is the most closely the title quote
    if do_normalize:
        outs = F.normalize(outs, dim=1)
    
    title_embedding = outs[:batch_size]
    sentences_embeddings = outs[batch_size:]
    s_similar = []

    for b in range(len(body_len)):
        i = body_len[b]
        t_embedding = title_embedding[b]
        s_embedding = sentences_embeddings[:i]
        sentences_embeddings = sentences_embeddings[i:]    
        
        cos_scores = util.pytorch_cos_sim(t_embedding, s_embedding)[0]
        cos_scores = cos_scores.cpu()

        top_results = np.argpartition(-cos_scores, range(1))[0:1]
        s_similar.append(s_embedding[top_results])    
        
    s_similar = torch.cat(s_similar, dim=0)
    return title_embedding, s_similar