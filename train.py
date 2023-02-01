from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn
import torch
import argparse
import pandas as pd
import os
import datetime

from datasets import(
    create_data_loader,
    tuplify_with_device,
)

from loss_func import QuoteCSELoss
from models import Encoder
from utils import AverageMeter, set_seed




def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--seed", default=123, type=int, help="set seed") 
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--max_len", default=512, type=int, help="max length")     
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers")    
    parser.add_argument("--dimension_size", default=768, type=int, help="dimension size") 
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")     
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="learning rate") 
    parser.add_argument("--weight_decay", default=1e-7, type=float, help="weight decay")   
    parser.add_argument("--epochs", default=500, type=int, help="epoch")   
    parser.add_argument("--iteration", default=1043200, type=int, help="data iteration")   
    parser.add_argument("--temperature", default=0.05, type=float, help="temperature")   
    
    parser.add_argument("--MODEL_DIR", default='./model/', type=str, help="where to save the trained model") 
    parser.add_argument("--DATA_PATH", default='./data/pretrain_data.pkl', type=str, help="data for pretraining")    

    args = parser.parse_args()

    if not os.path.exists(args.MODEL_DIR):
        os.makedirs(args.MODEL_DIR)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['WANDB_CONSOLE'] = 'off'
    set_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    args.backbone_model = BertModel.from_pretrained('skt/kobert-base-v1')
    args.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    loss_func = QuoteCSELoss(temperature=args.temperature, batch_size=args.batch_size)
    
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(args.device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    print('Making Dataloader')
    df = pd.read_pickle(args.DATA_PATH)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=args.seed)
    
    train_data_loader = create_data_loader(args,
                                           df = train_df, 
                                           shuffle = True,
                                           drop_last = True)
    valid_data_loader = create_data_loader(args,
                                           df = valid_df, 
                                           shuffle = True,
                                           drop_last = True)

    
    # train    
    print('Start Training')

    loss_data = []
    d_iter = 0
    stop = False

    for epoch in range(args.epochs):
        if stop == True:
            break

        losses = AverageMeter()

        valid_loss = []

        tbar1 = tqdm(train_data_loader)
        tbar2 = tqdm(valid_data_loader)

        encoder.train()
        for batch_idx, batch in enumerate(tbar1):
            if d_iter >= args.iteration:
                stop = True
                break

            d_iter += args.batch_size

            org_input_ids, org_attention_mask, \
            pos_input_ids, pos_attention_mask, \
            neg_input_ids, neg_attention_mask = tuplify_with_device(batch, args.device)
            outputs = encoder(
                input_ids=torch.cat([org_input_ids, pos_input_ids, neg_input_ids]),
                attention_mask=torch.cat([org_attention_mask, pos_attention_mask, neg_attention_mask])
            )

            loss = loss_func(outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), args.batch_size)
            tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)

            del org_input_ids, org_attention_mask, pos_input_ids, pos_attention_mask,\
                neg_input_ids, neg_attention_mask, outputs, loss

        ts = datetime.datetime.now().timestamp()
        loss_data.append([epoch, losses.avg, 'Train', ts])

        if stop == True:
            break

        # valid
        with torch.no_grad():
            for batch_idx, batch in enumerate(tbar2):
                org_input_ids, org_attention_mask, \
                pos_input_ids, pos_attention_mask, \
                neg_input_ids, neg_attention_mask = tuplify_with_device(batch, args.device)

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
                MODEL_SAVE_PATH = args.MODEL_DIR + 'model_' + str(epoch) + '.bin'
                torch.save(encoder.state_dict(), MODEL_SAVE_PATH)


    torch.save(encoder.state_dict(), args.MODEL_DIR + 'QuoteCSE_model.bin')
    torch.save(encoder, args.MODEL_DIR + 'QuoteCSE_entire_model.bin')

    # save loss
    df_loss = pd.DataFrame(loss_data, columns=('Epoch', 'Loss', 'Type', 'Time'))
    df_loss.to_csv(args.MODEL_DIR + 'loss.csv', sep=',', index=False)

    
    
if __name__ == "__main__":
    main()
