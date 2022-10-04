import torch
import torch.nn.functional as F


class QuoteCSELoss():
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            out = F.normalize(out, dim=1)
        batch_size = int(out.shape[0] / 3)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        out_1, out_2, out_3 = out.split(bs, dim=0)  # (B,D), (B,D), (B,D)

        sim_matrix_pos = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / self.temperature)
        sim_matrix_neg = torch.exp(torch.mm(out_1, out_3.t().contiguous()) / self.temperature)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

        return loss