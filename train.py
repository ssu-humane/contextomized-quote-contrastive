from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer

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
from util import make_pair, AverageMeter, set_seed
from pytorchtools import EarlyStopping



def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--seed", default=123, type=int, help="set seed") 
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--max_len", default=512, type=int, help="max length")     
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers")    
    parser.add_argument("--dimension_size", default=768, type=int, help="dimension size") 
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")     
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="learning rate") 
    parser.add_argument("--weight_decay", default=1e-7, type=float, help="weight decay")   
    parser.add_argument("--epochs", default=50, type=int, help="epoch")   
    parser.add_argument("--static_epochs", default=14, type=int, help="epoch for static")   
    parser.add_argument("--dynamic_epochs", default=2, type=int, help="epoch for dynamic")  
    parser.add_argument("--temperature", default=0.05, type=float, help="temperature")   
    parser.add_argument("--assignment", default='static', type=str, help="assignment type")   

    
    parser.add_argument("--MODEL_DIR", default='./model/', type=str, help="where to save the trained model") 
    parser.add_argument("--MODIFIED_DATA_PATH", default='./data/modified_sample.pkl', type=str, help="data for pretraining")    
    parser.add_argument("--VERBATIM_DATA_PATH", default='./data/verbatim_sample.pkl', type=str, help="data for pretraining")    

    args = parser.parse_args()

    if not os.path.exists(args.MODEL_DIR):
        os.makedirs(args.MODEL_DIR)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['WANDB_CONSOLE'] = 'off'
    set_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    args.backbone_model = get_kobert_model()
    args.tokenizer = get_tokenizer()
    loss_func = QuoteCSELoss(temperature=args.temperature, batch_size=args.batch_size)
    
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(args.device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    print('Making Dataloader')
    modified_df = pd.read_pickle(args.MODIFIED_DATA_PATH)
    verbatim_df = pd.read_pickle(args.VERBATIM_DATA_PATH)
  
    modified_df['title_quote'] = modified_df['title_quote'].map(lambda x:x[0])
    verbatim_df['title_quote'] = verbatim_df['title_quote'].map(lambda x:x[0])
    
    train_modified_df, test_modified_df = train_test_split(modified_df, test_size=0.2, random_state=args.seed)
    valid_modified_df, test_modified_df = train_test_split(test_modified_df, test_size=0.5, random_state=args.seed)

    train_verbatim_df, test_verbatim_df = train_test_split(verbatim_df, test_size=0.2, random_state=args.seed)
    valid_verbatim_df, test_verbatim_df = train_test_split(test_verbatim_df, test_size=0.5, random_state=args.seed)
    
    train_df = pd.concat([train_modified_df, train_verbatim_df])
    valid_df = pd.concat([valid_modified_df, valid_verbatim_df])
    test_df = pd.concat([test_modified_df, test_verbatim_df])
    
    train_data_loader = create_data_loader(args,
                                           df = train_df, 
                                           shuffle = True,
                                           drop_last = True)
    valid_data_loader = create_data_loader(args,
                                           df = valid_df, 
                                           shuffle = True,
                                           drop_last = True)

    
    early_stopping = EarlyStopping(patience = 3, verbose = True, path=args.MODEL_DIR + 'checkpoint_static_dynamic_early.bin')
    
    # train    
    print('Start Training')

    loss_data = []
    stop = False
    
    encoder.train()
    for epoch in range(args.epochs):
        if epoch >= args.static_epochs:
          args.assignment = 'dynamic'

        if epoch >= args.dynamic_epochs + args.static_epochs:
          break
        
        losses = AverageMeter()
        valid_loss = []
        
        tbar1 = tqdm(train_data_loader)
        tbar2 = tqdm(valid_data_loader)
        
        for title, body, body_len, pos_idx, neg_idx in tbar1:
          
          title_id, title_at = title['input_ids'].to(args.device).long(), title['attention_mask'].to(args.device).long()
          b_ids = []
          b_atts = []

          for b in range(len(body_len)):
            i = body_len[b]
            b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
            b_ids.append(b_id)
            b_atts.append(b_at)
          body_ids = torch.cat(b_ids, dim=0)
          body_atts = torch.cat(b_atts, dim=0)

          if args.assignment == 'static':
            pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx, neg_idx)
          
          elif args.assignment == 'dynamic':
            pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder)

          del body_ids, body_atts, body_len

          outputs = encoder(
            input_ids = torch.cat([title_id, pos_body_ids, neg_body_ids]),
            attention_mask = torch.cat([title_at, pos_body_atts, neg_body_atts]),
          )

          loss = loss_func(outputs)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          losses.update(loss.item(), args.batch_size)
          tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)

          del title_id, pos_body_ids, neg_body_ids, title_at, pos_body_atts, neg_body_atts, outputs, loss

        ts = datetime.datetime.now().timestamp()
        loss_data.append([epoch, losses.avg, 'Train', ts])


        # valid
        with torch.no_grad():
          for title, body, body_len, pos_idx, neg_idx in tbar2:
            title_id, title_at = title['input_ids'].to(args.device).long(), title['attention_mask'].to(args.device).long()
            b_ids = []
            b_atts = []

            for b in range(len(body_len)):
              i = body_len[b]
              b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
              b_ids.append(b_id)
              b_atts.append(b_at)
            body_ids = torch.cat(b_ids, dim=0)
            body_atts = torch.cat(b_atts, dim=0)

            if args.assignment == 'static':
              pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx, neg_idx)

            elif args.assignment == 'dynamic':
              pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder)

            del body_ids, body_atts, body_len

            outputs = encoder(
              input_ids = torch.cat([title_id, pos_body_ids, neg_body_ids]),
              attention_mask = torch.cat([title_at, pos_body_atts, neg_body_atts]),
            )

            loss = loss_func(outputs)
            valid_loss.append(loss.item())

            del title_id, pos_body_ids, neg_body_ids, title_at, pos_body_atts, neg_body_atts, outputs, loss

          avg_valid_loss = sum(valid_loss) / len(valid_loss)
          ts = datetime.datetime.now().timestamp()
          loss_data.append([epoch, avg_valid_loss, 'Valid', ts])

          print(str(epoch), 'th epoch, Avg Valid Loss: ', str(avg_valid_loss))

          early_stopping(avg_valid_loss, encoder) 
          if early_stopping.early_stop:
            break

    torch.save(encoder.state_dict(), args.MODEL_DIR + 'checkpoint.bin')

    # save loss
    df_loss = pd.DataFrame(loss_data, columns=('Epoch', 'Loss', 'Type', 'Time'))
    df_loss.to_csv(args.MODEL_DIR + 'loss.csv', sep=',', index=False)

    
    
if __name__ == "__main__":
    main()
