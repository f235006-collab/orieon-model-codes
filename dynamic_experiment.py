import copy
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from common import ROOT, RESULTS, metric_dict, save_predictions, update_metrics

SEED=42; LOOKBACK=20; MAX_EPOCHS=60; PATIENCE=10; BATCH_SIZE=64; WIDTH=48
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
protocol=json.load(open(RESULTS/'data_protocol.json'))
data=pd.read_csv(ROOT/'processed_data.csv',parse_dates=['Date'])
nt,nv=protocol['n_train'],protocol['n_val']
selection=json.load(open(RESULTS/'selected_temporal.json'))
temporal=selection['selected_temporal']
assert temporal in ['GRU','LSTM','MLP']

class GraphTemporal(nn.Module):
    def __init__(self,n_features,kind,dynamic=True):
        super().__init__(); self.kind=kind; self.dynamic=dynamic; self.n=n_features
        self.node_embedding=nn.Parameter(torch.randn(self.n,WIDTH)*0.1)
        self.value_proj=nn.Linear(1,WIDTH,bias=False)
        self.q=nn.Linear(WIDTH,WIDTH); self.k=nn.Linear(WIDTH,WIDTH)
        self.static_logits=nn.Parameter(torch.randn(self.n,self.n)*0.02)
        self.graph_proj=nn.Linear(WIDTH,WIDTH); self.self_proj=nn.Linear(WIDTH,WIDTH)
        self.readout=nn.Sequential(nn.Linear(2*WIDTH,WIDTH),nn.Tanh())
        if kind in ['GRU','LSTM']:
            self.rnn=(nn.GRU if kind=='GRU' else nn.LSTM)(WIDTH,64,batch_first=True)
        else:
            self.flat=nn.Sequential(nn.Flatten(),nn.Linear(LOOKBACK*WIDTH,64),nn.ReLU())
        self.head=nn.Sequential(nn.Dropout(0.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x,return_graph=False):
        b,t,n=x.shape
        h=torch.tanh(self.value_proj(x.unsqueeze(-1))+self.node_embedding.view(1,1,n,WIDTH))
        if self.dynamic:
            q=self.q(h); k=self.k(h)
            logits=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(WIDTH)
            eye=torch.eye(n,device=x.device).view(1,1,n,n)
            adj=torch.softmax(logits+eye,dim=-1)
        else:
            adj=torch.softmax(self.static_logits+torch.eye(n,device=x.device),dim=-1).view(1,1,n,n).expand(b,t,n,n)
        z=torch.relu(self.graph_proj(torch.matmul(adj,h))+self.self_proj(h))
        pooled=torch.cat([z.mean(dim=2),z[:,:,3,:]],dim=-1)  # node 3 is Gold_Close in the supplied schema
        seq=self.readout(pooled)
        if self.kind in ['GRU','LSTM']: encoded=self.rnn(seq)[0][:,-1,:]
        else: encoded=self.flat(seq)
        out=self.head(encoded)
        return (out,adj) if return_graph else out


def run_experiment(name,feature_cols,dynamic=True):
    Xraw=data[feature_cols].values.astype(np.float32); yraw=data.y_close_t1.values.astype(np.float32)
    xs=StandardScaler().fit(Xraw[:nt]); ys=StandardScaler().fit(yraw[:nt].reshape(-1,1))
    X=xs.transform(Xraw).astype(np.float32); y=ys.transform(yraw.reshape(-1,1)).astype(np.float32).ravel()
    def idx(split):
        if split=='train': return np.arange(LOOKBACK-1,nt)
        if split=='val': return np.arange(nt,nt+nv)
        return np.arange(nt+nv,len(data))
    loaders={}
    for split in ['train','val','test']:
        ii=idx(split)
        seq=np.stack([X[i-LOOKBACK+1:i+1] for i in ii])
        ds=TensorDataset(torch.from_numpy(seq),torch.from_numpy(y[ii]).reshape(-1,1))
        loaders[split]=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=(split=='train'))
    torch.manual_seed(SEED); np.random.seed(SEED)
    model=GraphTemporal(len(feature_cols),temporal,dynamic)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5); loss_fn=nn.MSELoss()
    best_loss=float('inf'); best=None; best_epoch=0; stale=0; history=[]
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total=0; count=0
        for xb,yb in loaders['train']:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()*len(xb); count+=len(xb)
        model.eval(); vl=0; vc=0
        with torch.no_grad():
            for xb,yb in loaders['val']:
                loss=loss_fn(model(xb),yb); vl+=loss.item()*len(xb); vc+=len(xb)
        history.append({'epoch':epoch,'train_mse_scaled':total/count,'val_mse_scaled':vl/vc})
        if epoch==1 or epoch%5==0: print(name,'epoch',epoch,'train',total/count,'val',vl/vc,flush=True)
        if vl/vc<best_loss-1e-6:
            best_loss=vl/vc; best=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else:
            stale+=1
            if stale>=PATIENCE: break
    model.load_state_dict(best); torch.save(model.state_dict(),RESULTS/f'{name}_best.pt')
    def predict(split):
        model.eval(); out=[]
        with torch.no_grad():
            for xb,_ in loaders[split]: out.append(model(xb).numpy())
        p=np.concatenate(out).ravel(); return ys.inverse_transform(p.reshape(-1,1)).ravel()
    for split in ['val','test']:
        ii=idx(split); frame=data.iloc[ii]; p=predict(split)
        m=metric_dict(frame.y_close_t1.values,p,frame.Gold_Close.values)
        update_metrics(name,split,m)
        save_predictions(name,split,frame.Date.dt.strftime('%Y-%m-%d'),frame.y_close_t1.values,p,frame.Gold_Close.values)
        print(name,split,json.dumps(m),flush=True)
    params=sum(p.numel() for p in model.parameters())
    with open(RESULTS/f'{name}_training.json','w') as f:
        json.dump({'best_epoch':best_epoch,'history':history,'parameters':params,
                   'dynamic_adjacency':dynamic,'temporal':temporal,'features':feature_cols},f,indent=2)
    return model

if __name__ == '__main__':
    import sys
    requested = sys.argv[1] if len(sys.argv) > 1 else 'all'
    features=protocol['features']
    if requested in ['all','dynamic']:
        start=time.time(); run_experiment('DGNN-GRU',features,True)
        print('Dynamic seconds',time.time()-start,flush=True)
    if requested in ['all','static']:
        start=time.time(); run_experiment('SGNN-GRU',features,False)
        print('Static seconds',time.time()-start,flush=True)
    if requested in ['all','nosentiment']:
        no_sent=[c for c in features if not any(s in c for s in ['Sentiment','News_','Positive_Ratio','Negative_Ratio'])]
        start=time.time(); run_experiment('DGNN-GRU-NoSentiment',no_sent,True)
        print('No-sentiment seconds',time.time()-start,flush=True)
    print('REQUESTED DYNAMIC EXPERIMENTS COMPLETE',flush=True)
