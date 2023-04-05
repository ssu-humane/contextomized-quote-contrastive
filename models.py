import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.encoder = args.backbone_model
        self.mlp_projection = nn.Sequential(nn.Linear(args.dimension_size, args.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(args.hidden_size, args.hidden_size, bias=True))

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
        embedding = output['pooler_output']
        return self.mlp_projection(embedding)
    
    
    
class Detection_Model(nn.Module):
    def __init__(self, num_cls, args):
        super(Detection_Model, self).__init__()
        self.dim = args.classifier_input_size*4
        self.hidden = args.classifier_hidden_size
        self.out = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, embedding):
        return self.out(embedding)
