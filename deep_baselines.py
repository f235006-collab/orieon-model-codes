import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from common import ROOT, RESULTS, metric_dict, save_predictions, update_metrics

SEED = 42
LOOKBACK = 20
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

protocol = json.load(open(RESULTS / 'data_protocol.json'))
data = pd.read_csv(ROOT / 'processed_data.csv', parse_dates=['Date'])
features = protocol['features']
nt, nv = protocol['n_train'], protocol['n_val']
Xraw = data[features].values.astype(np.float32)
yraw = data.y_close_t1.values.astype(np.float32)
x_scaler = StandardScaler().fit(Xraw[:nt])
y_scaler = StandardScaler().fit(yraw[:nt].reshape(-1, 1))
X = x_scaler.transform(Xraw).astype(np.float32)
y = y_scaler.transform(yraw.reshape(-1, 1)).astype(np.float32).ravel()

def indices_for(split):
    if split == 'train': return np.arange(LOOKBACK-1, nt)
    if split == 'val': return np.arange(nt, nt+nv)
    return np.arange(nt+nv, len(data))

def make_arrays(indices):
    seq = np.stack([X[i-LOOKBACK+1:i+1] for i in indices])
    return torch.from_numpy(seq), torch.from_numpy(y[indices]).reshape(-1,1)

loaders = {}
for split in ['train','val','test']:
    idx = indices_for(split)
    xs, ys = make_arrays(idx)
    loaders[split] = DataLoader(TensorDataset(xs,ys), batch_size=BATCH_SIZE,
                                shuffle=(split=='train'), drop_last=False)

class SequenceModel(nn.Module):
    def __init__(self, kind, n_features):
        super().__init__()
        self.kind = kind
        if kind == 'MLP':
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(LOOKBACK*n_features,128),
                                         nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,64), nn.ReLU())
        elif kind in ['LSTM','GRU']:
            self.rnn = (nn.LSTM if kind=='LSTM' else nn.GRU)(n_features,64,batch_first=True)
        else: raise ValueError(kind)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        if self.kind=='MLP': h=self.encoder(x)
        else: h=self.rnn(x)[0][:,-1,:]
        return self.head(h)

def predict(model,loader):
    model.eval(); out=[]
    with torch.no_grad():
        for xb,_ in loader: out.append(model(xb).numpy())
    p=np.concatenate(out).ravel()
    return y_scaler.inverse_transform(p.reshape(-1,1)).ravel()

def train_model(kind):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model=SequenceModel(kind,len(features))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    loss_fn=nn.MSELoss()
    best_loss=float('inf'); best_state=None; best_epoch=0; stale=0; history=[]
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total=0.; count=0
        for xb,yb in loaders['train']:
            opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()*len(xb); count+=len(xb)
        model.eval(); vtotal=0.; vcount=0
        with torch.no_grad():
            for xb,yb in loaders['val']:
                vloss=loss_fn(model(xb),yb); vtotal+=vloss.item()*len(xb); vcount+=len(xb)
        train_loss=total/count; val_loss=vtotal/vcount
        history.append({'epoch':epoch,'train_mse_scaled':train_loss,'val_mse_scaled':val_loss})
        if epoch==1 or epoch%5==0: print(kind,'epoch',epoch,'train',train_loss,'val',val_loss,flush=True)
        if val_loss < best_loss-1e-6:
            best_loss=val_loss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else:
            stale+=1
            if stale>=PATIENCE: break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(),RESULTS/f'{kind}_best.pt')
    return model,best_epoch,history

summary={}
for kind in ['MLP','LSTM','GRU']:
    start=time.time(); model,best_epoch,history=train_model(kind)
    summary[kind]={'best_epoch':best_epoch,'history':history}
    for split in ['val','test']:
        idx=indices_for(split); frame=data.iloc[idx]
        p=predict(model,loaders[split])
        m=metric_dict(frame.y_close_t1.values,p,frame.Gold_Close.values)
        update_metrics(kind,split,m)
        save_predictions(kind,split,frame.Date.dt.strftime('%Y-%m-%d'),frame.y_close_t1.values,p,frame.Gold_Close.values)
        print(kind,split,json.dumps(m),flush=True)
    print(kind,'seconds',time.time()-start,flush=True)

# Selection is based only on validation RMSE. MLP remains eligible as the hybrid temporal reader.
selection=min(['MLP','LSTM','GRU'],key=lambda k: next(r for r in summary[k]['history'] if r['epoch']==summary[k]['best_epoch'])['val_mse_scaled'])
with open(RESULTS/'deep_training.json','w') as f: json.dump(summary,f,indent=2)
with open(RESULTS/'selected_temporal.json','w') as f:
    json.dump({'selected_temporal':selection,'criterion':'minimum scaled validation MSE at early-stopping checkpoint',
               'lookback':LOOKBACK,'seed':SEED,'feature_scaler':'StandardScaler fitted on training rows only'},f,indent=2)
print('SELECTED TEMPORAL:',selection,flush=True)
