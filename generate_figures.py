import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import ROOT, RESULTS, FIGURES

protocol=json.load(open(RESULTS/'data_protocol.json'))
data=pd.read_csv(ROOT/'processed_data.csv',parse_dates=['Date'])
metrics=pd.read_csv(RESULTS/'metrics.csv')
news=pd.read_csv(ROOT/'filtered_news.csv',parse_dates=['Date'])
colors={'dark':'#263238','teal':'#38726C','gold':'#B08A3E','rust':'#9A5B4F','gray':'#8A9499','light':'#DCE5E5'}
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':140})

# Figure 1: observed series and chronological partitions.
fig,ax=plt.subplots(figsize=(10,4.6))
nt,nv=protocol['n_train'],protocol['n_val']
ax.plot(data.Date,data.Gold_Close,color=colors['dark'],lw=1.15)
ax.axvspan(data.Date.iloc[0],data.Date.iloc[nt-1],color=colors['light'],alpha=.55,label='Training')
ax.axvspan(data.Date.iloc[nt],data.Date.iloc[nt+nv-1],color=colors['gold'],alpha=.16,label='Validation')
ax.axvspan(data.Date.iloc[nt+nv],data.Date.iloc[-1],color=colors['teal'],alpha=.12,label='Test')
ax.set_title('The held-out test period contains a pronounced upward price regime shift',loc='left',fontweight='bold')
ax.set_ylabel('Gold closing price (source units)'); ax.set_xlabel('Date')
ax.legend(frameon=False,ncol=3,loc='upper left'); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(FIGURES/'figure1_regime_shift.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 2: validation and test RMSE for every estimated model.
selected_classical=json.load(open(RESULTS/'selected_classical.json'))
arima_name=f"ARIMA{tuple(selected_classical['ARIMA_order'])}"
order=['Persistence','Ridge',arima_name,'RandomForest','LightGBM','MLP','LSTM','GRU','SGNN-GRU','DGNN-GRU','DGNN-GRU-NoSentiment']
labels={m:m for m in order}; labels['DGNN-GRU-NoSentiment']='DGNN–GRU (no sentiment)'
fig,axes=plt.subplots(1,2,figsize=(11,6.2))
for ax,split,title in zip(axes,['val','test'],['Ridge narrowly leads the validation period','Persistence leads the held-out test period']):
    tab=metrics[metrics.Split==split].set_index('Model').reindex(order).reset_index()
    y=np.arange(len(tab))[::-1]
    barcolors=[colors['teal'] if m in ['Persistence','Ridge'] else colors['gold'] if 'DGNN' in m or 'SGNN' in m else colors['gray'] for m in tab.Model]
    ax.barh(y,tab.RMSE,color=barcolors,height=.68)
    ax.set_yticks(y, [labels[m] for m in tab.Model]); ax.invert_yaxis()
    ax.set_xlabel('RMSE (source price units)'); ax.set_title(title,loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.2)
    for yy,v in zip(y,tab.RMSE): ax.text(v+tab.RMSE.max()*.012,yy,f'{v:,.1f}',va='center',fontsize=8)
    ax.set_xlim(0,tab.RMSE.max()*1.18)
fig.tight_layout(); fig.savefig(FIGURES/'figure2_rmse_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 3: test trajectories for the strongest simple models and selected neural models.
def pred(model): return pd.read_csv(RESULTS/f'{model}_test_predictions.csv',parse_dates=['Date'])
base=pred('Persistence'); ridge=pred('Ridge'); dgnn=pred('DGNN-GRU'); gru=pred('GRU')
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
axes[0].plot(base.Date,base.actual,color=colors['dark'],lw=1.4,label='Actual next-day close')
axes[0].plot(base.Date,base.predicted,color=colors['teal'],lw=1.0,label='Persistence')
axes[0].plot(ridge.Date,ridge.predicted,color=colors['gold'],lw=1.0,label='Ridge')
axes[0].set_title('Simple models track the late-sample price level more closely',loc='left',fontweight='bold')
axes[1].plot(base.Date,base.actual,color=colors['dark'],lw=1.4,label='Actual next-day close')
axes[1].plot(dgnn.Date,dgnn.predicted,color=colors['rust'],lw=1.0,label='Dynamic GNN–GRU')
axes[1].plot(gru.Date,gru.predicted,color=colors['gray'],lw=1.0,label='GRU')
axes[1].set_title('Direct-price recurrent models do not extrapolate the rally',loc='left',fontweight='bold')
for ax in axes: ax.set_ylabel('Price (source units)'); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,ncol=3,loc='upper left')
axes[1].set_xlabel('Test date')
fig.tight_layout(); fig.savefig(FIGURES/'figure3_test_predictions.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 4: sentiment score distribution after the fixed relevance filter.
fig,ax=plt.subplots(figsize=(8.5,4.3))
ax.hist(news.Score,bins=45,color=colors['teal'],alpha=.85,edgecolor='white',linewidth=.3)
ax.axvline(news.Score.mean(),color=colors['rust'],ls='--',lw=1.4,label=f'Mean = {news.Score.mean():.3f}')
ax.set_title('FinBERT scores vary widely across the filtered headline sample',loc='left',fontweight='bold')
ax.set_xlabel('Supplied FinBERT sentiment score'); ax.set_ylabel('Number of headlines')
ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(FIGURES/'figure4_sentiment_distribution.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 5: schematic of the implemented architecture, not a claim of published architecture.
fig,ax=plt.subplots(figsize=(11,3.6)); ax.set_xlim(0,11); ax.set_ylim(0,3.6); ax.axis('off')
boxes=[(.2,'20-day multimodal\nsequences'),(2.35,'Node identity +\nvalue embedding'),(4.5,'Time-specific adjacency\n$A_t=softmax(Q_tK_t^T/\\sqrt{d}+I)$'),(6.95,'Graph propagation +\nnode readout'),(9.1,'GRU and linear\nnext-close head')]
for x,text in boxes:
    w=1.75 if x<9 else 1.65
    patch=FancyBboxPatch((x,.95),w,1.45,boxstyle='round,pad=0.08,rounding_size=0.08',facecolor='#EEF3F3',edgecolor=colors['teal'],lw=1.2)
    ax.add_patch(patch); ax.text(x+w/2,1.675,text,ha='center',va='center',fontsize=9,color=colors['dark'])
for x1,x2 in [(1.95,2.35),(4.1,4.5),(6.65,6.95),(8.7,9.1)]:
    ax.add_patch(FancyArrowPatch((x1,1.68),(x2,1.68),arrowstyle='-|>',mutation_scale=13,color=colors['gold'],lw=1.3))
ax.text(.2,3.1,'Implemented dynamic graph–GRU architecture',fontsize=13,fontweight='bold',color=colors['dark'])
ax.text(.2,.35,'Technical indicators + macrofinancial variables + filtered FinBERT sentiment',fontsize=10,color=colors['gray'])
fig.tight_layout(); fig.savefig(FIGURES/'figure5_architecture.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Derived values used in the paper, calculated directly from saved outputs.
test=metrics[metrics.Split=='test'].set_index('Model'); val=metrics[metrics.Split=='val'].set_index('Model')
train_target_max=float(data.y_close_t1.iloc[:nt].max())
test_targets=data.y_close_t1.iloc[nt+nv:]
above_train_max=int((test_targets>train_target_max).sum())
summary={
 'test_targets_above_train_max':above_train_max,
 'test_targets_above_train_max_pct':above_train_max/len(test_targets)*100,
 'dgnn_vs_gru_test_rmse_reduction_pct':(test.loc['GRU','RMSE']-test.loc['DGNN-GRU','RMSE'])/test.loc['GRU','RMSE']*100,
 'dgnn_vs_gru_val_rmse_reduction_pct':(val.loc['GRU','RMSE']-val.loc['DGNN-GRU','RMSE'])/val.loc['GRU','RMSE']*100,
 'static_vs_dynamic_test_rmse_reduction_pct':(test.loc['DGNN-GRU','RMSE']-test.loc['SGNN-GRU','RMSE'])/test.loc['DGNN-GRU','RMSE']*100,
 'sentiment_full_vs_ablation_test_rmse_reduction_pct':(test.loc['DGNN-GRU-NoSentiment','RMSE']-test.loc['DGNN-GRU','RMSE'])/test.loc['DGNN-GRU-NoSentiment','RMSE']*100,
 'sentiment_full_vs_ablation_val_rmse_reduction_pct':(val.loc['DGNN-GRU-NoSentiment','RMSE']-val.loc['DGNN-GRU','RMSE'])/val.loc['DGNN-GRU-NoSentiment','RMSE']*100,
 'filtered_score_mean':float(news.Score.mean()),
 'filtered_score_std':float(news.Score.std()),
}
with open(RESULTS/'paper_summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print('Figures written to',FIGURES)
