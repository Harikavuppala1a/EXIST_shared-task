import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda,nn
import sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.autograd import Variable


XLMRTokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
XLMRModel = AutoModel.from_pretrained(sys.argv[1])

device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv('./train_'+str(sys.argv[2])+".csv", sep=',', names=['abstract','type_id'])
test_df = pd.read_csv('./validation_'+str(sys.argv[2])+".csv", sep=',', names=['abstract','type_id'])

MAX_LEN = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 6
EPOCHS = 10
ATT_DIM = 300
LEARNING_RATE = 1e-05
tokenizer = XLMRTokenizer


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        abstract = str(self.data.abstract[index])
        abstract = " ".join(abstract.split())
        inputs = self.tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.type_id[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

train_dataset=df
testing_dataset=test_df

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class XLMRClass(torch.nn.Module):
    def __init__(self):
        super(XLMRClass, self).__init__()
        self.l1 = XLMRModel
        self.lstm = nn.LSTM(768,ATT_DIM,num_layers=1,bidirectional=True,batch_first=True)
        self.pre_classifier = torch.nn.Linear(ATT_DIM*2, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)
        # self.fc_att = nn.Linear(1536, 1)
        self.attention_layer = Attention(ATT_DIM*2, 512)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(2, 512, ATT_DIM)),Variable(torch.zeros(2, 512, ATT_DIM)))
        return h.to(device), c.to(device)

    def forward(self, input_ids, attention_mask):
        h_0, c_0 = self.init_hidden(TRAIN_BATCH_SIZE)

        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_t = output_1[0]
        hidden_state = hidden_state_t.permute(1,0,2)
        
        output_lstm, (h_n, c_n) = self.lstm(hidden_state, (h_0, c_0))
        output_permuted = output_lstm.permute(1,0,2) 
        h_lstm_atten = self.attention_layer(output_permuted)
        # print(h_lstm_atten.shape)
        # exit()
        # att = self.fc_att(output_permuted).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        # r_att = torch.sum(att.unsqueeze(-1) * output_permuted, dim=1)  # [b,h*2]

        pooler = self.pre_classifier(h_lstm_atten)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # exit()
        # exit()
        return output

model = XLMRClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}\n")
    print(f"Training Accuracy Epoch: {epoch_accu}\n")
    print("\n")
    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; tr_loss = 0
    nb_tr_steps =0
    nb_tr_examples =0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}\n")
    print(f"Validation Accuracy Epoch: {epoch_accu}\n")
    
    return epoch_accu

for epoch in range(EPOCHS):
    train(epoch)
    acc = valid(model, testing_loader)
    print("\n")
    print("\n")



