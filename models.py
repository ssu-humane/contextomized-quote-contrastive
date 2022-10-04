from kobert_transformers import get_kobert_model
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.encoder = get_kobert_model()
        self.mlp_projection = nn.Sequential(nn.Linear(config.dimension_size, config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(config.hidden_size, config.hidden_size, bias=True))

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
        embedding = output['pooler_output']

        return self.mlp_projection(embedding)
