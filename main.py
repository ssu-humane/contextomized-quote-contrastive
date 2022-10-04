from tqdm.notebook import tqdm
import torch.nn as nn
import torch

import pandas as pd
import os
import datetime

from models import Model
from preprocessing import make_dataset
from utils import (
    create_data_loader,
    set_seed,
    AverageMeter,
    tuplify_with_device,
)
from config import ModelArguments, DataArguments
from loss_func import QuoteCSELoss


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['WANDB_CONSOLE'] = 'off'

    set_seed(ModelArguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = ModelArguments.batch_size * torch.cuda.device_count()
    loss_func = QuoteCSELoss(temperature=ModelArguments.temperature, batch_size=batch_size)

    encoder = Model(ModelArguments)
    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=ModelArguments.learning_rate, weight_decay=ModelArguments.weight_decay)
    optimizer.zero_grad()

    train_df, valid_df, test_df = make_dataset()
    train_data_loader = create_data_loader(df=train_df, MAX_LEN=ModelArguments.MAX_LEN,
                                           batch_size=batch_size,
                                           num_workers = ModelArguments.num_workers,
                                           shuffle=DataArguments.DATA_LOADER_SHUFFLE,
                                           drop_last=DataArguments.DATA_LOADER_DROP_LAST)
    valid_data_loader = create_data_loader(df=valid_df, MAX_LEN=ModelArguments.MAX_LEN,
                                           batch_size=batch_size,
                                           num_workers = ModelArguments.num_workers,
                                           shuffle=DataArguments.DATA_LOADER_SHUFFLE,
                                           drop_last=DataArguments.DATA_LOADER_DROP_LAST)

    print('Start Training')

    loss_data = []
    d_iter = 0
    stop = False

    for epoch in range(ModelArguments.epochs):
        if stop == True:
            break

        losses = AverageMeter()

        valid_loss = []

        tbar1 = tqdm(train_data_loader)
        tbar2 = tqdm(valid_data_loader)

        encoder.train()
        for batch_idx, batch in enumerate(tbar1):
            if d_iter >= ModelArguments.iteration:
                stop = True
                break

            d_iter += batch_size

            org_input_ids, org_attention_mask, \
            pos_input_ids, pos_attention_mask, \
            neg_input_ids, neg_attention_mask = tuplify_with_device(batch, device)
            outputs = encoder(
                input_ids=torch.cat([org_input_ids, pos_input_ids, neg_input_ids]),
                attention_mask=torch.cat([org_attention_mask, pos_attention_mask, neg_attention_mask])
            )

            loss = loss_func(outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch_size)
            tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)

            del org_input_ids, org_attention_mask, pos_input_ids, pos_attention_mask,\
                neg_input_ids, neg_attention_mask, outputs, loss

        ts = datetime.datetime.now().timestamp()
        loss_data.append([epoch, losses.avg, 'Train', ts])

        if stop == True:
            break

        encoder.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tbar2):
                org_input_ids, org_attention_mask, \
                pos_input_ids, pos_attention_mask, \
                neg_input_ids, neg_attention_mask = tuplify_with_device(batch, device)

                outputs = encoder(
                    input_ids=torch.cat([org_input_ids, pos_input_ids, neg_input_ids]),
                    attention_mask=torch.cat([org_attention_mask, pos_attention_mask, neg_attention_mask])
                )

                loss = loss_func(outputs)
                valid_loss.append(loss.item())

                del org_input_ids, org_attention_mask, pos_input_ids, pos_attention_mask,\
                    neg_input_ids, neg_attention_mask, outputs, loss

            avg_valid_loss = sum(valid_loss) / len(valid_loss)
            ts = datetime.datetime.now().timestamp()
            loss_data.append([epoch, avg_valid_loss, 'Valid', ts])

            print(str(epoch), 'th epoch, Avg Valid Loss: ', str(avg_valid_loss), 'd_iter:', d_iter)

            if epoch % 10 == 0:
                MODEL_SAVE_PATH = ModelArguments.MODEL_DIR + 'model_' + str(epoch) + '.bin'
                torch.save(encoder.state_dict(), MODEL_SAVE_PATH)


    torch.save(encoder.state_dict(), ModelArguments.MODEL_DIR + 'final_model.bin')
    torch.save(encoder, ModelArguments.MODEL_DIR + 'final_entire_model.bin')

    torch.save(encoder.state_dict(), ModelArguments.MODEL_DIR + 'final_model.pt')
    torch.save(encoder, ModelArguments.MODEL_DIR + 'final_entire_model.pt')

    # save loss
    df_loss = pd.DataFrame(loss_data, columns=('Epoch', 'Loss', 'Type', 'Time'))
    df_loss.to_csv(ModelArguments.MODEL_DIR + 'loss.csv', sep=',', index=False)


if __name__ == "__main__":
    main()