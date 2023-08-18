import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import os
from tqdm.notebook import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split
from module.util import BERTClassifier, BERTDataset, load_data, calc_accuracy

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
PATH = './data.csv'
max_len = 100
batch_size = 64
warmup_ratio = 0.1
num_epochs = 7
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

def train():
    data = load_data(PATH)
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    
    data_list = []
    for q, label in zip(data['review'], data['class'])  :
        data = []
        data.append(q)
        data.append(str(label))

        data_list.append(data)
        
    dataset_train, dataset_test = train_test_split(data_list, test_size = 0.2, shuffle = True, random_state = 23)
    
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)
    
    # BERTDataset : 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding하는 함수
    data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)
    
    # torch 형식의 dataset을 만들어 입력 데이터셋의 전처리 마무리
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5)

    # 모델 정의
    model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(device)
    
    
    # optimizer와 schedule 설정
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
    # loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 loss function
    loss_fn = nn.BCELoss # 이진분류를 위한 loss function

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

    # 학습 시작
    train_history = []
    test_history = []
    loss_history = []

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            
            # print(label.shape, out.shape)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                train_history.append(train_acc / (batch_id+1))
                loss_history.append(loss.data.cpu().numpy())
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        # train_history.append(train_acc / (batch_id+1))

        
        # .eval() : nn.Module에서 train time과 eval time에서 수행하는 다른 작업을 수행할 수 있도록 switching 하는 함수
        # 즉, model이 Dropout이나 BatNorm2d를 사용하는 경우, train 시에는 사용하지만 evaluation을 할 때에는 사용하지 않도록 설정해주는 함수
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        test_history.append(test_acc / (batch_id+1))

    # 모델 저장
    MODEL_PATH = os.getcwd() + '/model_save'
    os.makedirs(MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH + '/kobert.pt')
        
        
if __name__ == "__main__":
    train()
