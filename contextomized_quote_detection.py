from tqdm.auto import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import math

from detection_datasets import(
    create_data_loader,
    make_tensorloader,
)

from models import Encoder, Detection_Model
from utils import AverageMeter, set_seed

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from transformers import BertModel, AdamW, get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer


def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--seed", default=0, type=int, help="set seed") 
    parser.add_argument("--split_seed", default=0, type=int, help="set seed") 
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")    
    parser.add_argument("--max_len", default=512, type=int, help="max length")     
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers")    
    parser.add_argument("--dimension_size", default=768, type=int, help="dimension size") 
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")    
    parser.add_argument("--classifier_input_size", default=100, type=int, help="input dimension size") 
    parser.add_argument("--classifier_hidden_size", default=64, type=int, help="hidden size")     
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="learning rate") 
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")   
    parser.add_argument("--epochs", default=10, type=int, help="epoch")    
    parser.add_argument("--schedule", default=True, type=bool, help="")    
    
    parser.add_argument("--DATA_DIR", default='./data/contextomized_detection_data.pkl', type=str, help="data to dectect contextomized quote") 
#    parser.add_argument("--MODEL_DIR", default='./model/QuoteCSE_model.bin', type=str, help="trained QuoteCSE model") 
    parser.add_argument("--MODEL_DIR", default='./model/model_0.bin', type=str, help="trained QuoteCSE model") 
    parser.add_argument("--MODEL_SAVE_DIR", default='./model/contextomized_detection/', type=str, help="trained QuoteCSE model")
    
    args = parser.parse_args()

    if not os.path.exists(args.MODEL_SAVE_DIR):
        os.makedirs(args.MODEL_SAVE_DIR)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['WANDB_CONSOLE'] = 'off'
    set_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    args.backbone_model = BertModel.from_pretrained('skt/kobert-base-v1')
    args.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    df = pd.read_pickle(args.DATA_DIR)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.split_seed, stratify=df['label'])
    
    ros = RandomOverSampler(random_state=args.seed)
    X_train, y_train = ros.fit_resample(X=df_train.loc[:, ['title_quote', 'sentence_quotes']].values, y=df_train['label'])
    df_train_ros = pd.DataFrame(X_train, columns=['title_quote', 'sentence_quotes'])
    df_train_ros['label'] = y_train
    
 
    
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder)
    encoder.load_state_dict(torch.load(args.MODEL_DIR))
    encoder = encoder.to(args.device)

    classifier = Detection_Model(2, args)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.to(args.device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    if args.schedule:
        total_steps = total_steps = math.ceil(len(df_train_ros) / args.batch_size) * args.epochs
        warmup_steps = math.ceil(len(df_train_ros) / args.batch_size)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = total_steps
        )
        
    
    print('Making Dataloader')
    train_data_loader = create_data_loader(args, df_train_ros, shuffle=True, drop_last=True)
    test_data_loader = create_data_loader(args, df_test, shuffle=False, drop_last=True)

    trainloader =  make_tensorloader(args, encoder, train_data_loader, train=True)
    testloader  = make_tensorloader(args, encoder, test_data_loader)


    loss_data = []
    print('Start Training')
    for epoch in range(args.epochs):
        train_losses = AverageMeter()

        train_loss = []

        tbar = tqdm(trainloader)
        classifier.train()
        for embedding, label in tbar:
            embedding = embedding.to(args.device)
            label = label.to(args.device)

            out = classifier(embedding)
            train_loss = loss_func(out, label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if args.schedule:
                scheduler.step()

            train_losses.update(train_loss.item(), args.batch_size)
            tbar.set_description("train_loss: {0:.4f}".format(train_losses.avg), refresh=True)

            del out, train_loss

        loss_data.append([epoch, train_losses.avg, 'Train'])


        tbar2 = tqdm(testloader)
        classifier.eval()
        with torch.no_grad():
            predictions = []
            answers = []
            for embedding, label in tbar2:
                out = classifier(embedding)
                preds = torch.argmax(out,dim=1)
                predictions.extend(preds)
                answers.extend(label)

                del out, preds

            predictions = torch.stack(predictions).cpu().tolist()
            answers = torch.stack(answers).cpu().tolist()

    torch.save(classifier.state_dict(), args.MODEL_SAVE_DIR + 'final_model.bin')
    torch.save(classifier, args.MODEL_SAVE_DIR + 'final_entire_model.bin')

    print('accuracy:', accuracy_score(answers, predictions))
    print('f1:', f1_score(answers, predictions, average='binary'))
    print('precision:', precision_score(answers, predictions))
    print('recall:', recall_score(answers, predictions))
    print('auc:', roc_auc_score(answers, predictions))
    
    
    
if __name__ == "__main__":
    main()