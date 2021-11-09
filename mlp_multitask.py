import json
import os
from collections import OrderedDict

import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import roc_auc_score,classification_report

import ray #https://github.com/ray-project/ray/issues/16013
# ray.init("auto")
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from utils import augment

ray.init( dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from modules import Net
from settings import result_path, data_path, log_dir, predictors,outcome1,outcome2,outcome3

device='cpu'
init=None
experiment=f"MLP"

#load data
maindir = data_path # Directory with your files
traincsv = os.path.join(maindir,"Train.csv")
testcsv = os.path.join(maindir,"Test.csv")

train = pd.read_csv(traincsv)
test =  pd.read_csv(testcsv)
X=train[predictors].values
X_test=test[predictors].values
label_enc1=LabelEncoder()
label_enc2=LabelEncoder()
label_enc3=LabelEncoder()
y1=label_enc1.fit_transform(train[outcome1])
y2=label_enc1.fit_transform(train[outcome2])
y3=label_enc1.fit_transform(train[outcome3])
y=np.stack([y1,y2,y2],axis=1)



configs={
    'lr':tune.loguniform(0.0001,0.01),
    'l2': tune.loguniform(0.01,0.1),
    'batch_size' : tune.choice([64,128,256,]),
    # 'pos_weight': tune.choice([1.0,2.0,5.0,10.0,20.0,50.0]),
    'l1_sparsity' : tune.loguniform(0.01,0.1),
    'gradient_norm':tune.choice([0.00001,0.0001,0.001,0.01,0.1,1.0,2.0]),
    # 'momentum':tune.choice([0.0,0.9,0.99]),
    'augment_sd':tune.choice([0.001,0.01,0.1,1.0]),

}
config={i:v.sample() for i,v in configs.items()}

def get_model():
    if device == 'cpu':
        torch.manual_seed(25)
    else:
        torch.cuda.manual_seed(25)

    model=Net(dim_input=X.shape[1],n_hidden_layers=2,
              dim_hidden_layers=16,dim_out=1,multitask=True,n_hidden_multitask=0)
    if init is not None:
        init_weights_path=os.path.join(result_path,f"weights/{init}-weights.pth")
        _,init_model_state=torch.load(init_weights_path)
        enc_state=OrderedDict()
        for k,v in init_model_state.items():
            if k =='encoder.fc.weight':
                break
            enc_state[k.replace('encoder.','')]=v
        incompatible=model.load_state_dict(enc_state,strict=False)
        for k in incompatible.missing_keys:
            if k not in ('fc.weight','fc.bias'):
                raise Exception(f"Model {k} not in model")
        if len(incompatible.unexpected_keys)>0:
            raise Exception(f'Unexpected keys during model initialization...')

    return model

def get_optimizer(config,model):
    optimizer=Adam(params=model.parameters(),
                  lr=config['lr'],weight_decay=config['l2'],
                  # momentum=config['momentum'],
                   )
    return optimizer

def get_train_loader(config,X_train,y_train):
    dataset=TensorDataset(torch.tensor(augment(X_train,sd=config['augment_sd'])),
                          torch.tensor(y_train,dtype=torch.float))

    train_loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    return train_loader




def get_val_loader(X_valid,y_valid=None,batch_size=256,):
    if y_valid is not None:
        dataset = TensorDataset(torch.tensor(augment(X_valid,sd=0.001)), torch.tensor(y_valid,dtype=torch.float))
    else:
        dataset = TensorDataset(torch.tensor(X_valid), )
    val_loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    return val_loader

class MultiTaskLoss(nn.Module):
    def __init__(self,device):
        super(MultiTaskLoss, self).__init__()
        self.loss_fn = [nn.CrossEntropyLoss(reduction='mean').to(device) for _ in range(3)]
        self.eta = nn.Parameter(torch.zeros(3, device=device))

    def forward(self, preds,target):
        loss = [self.loss_fn[i](preds[i],target[:,i]) for i in range(3)]

        total_loss = torch.stack(loss) * torch.exp(-self.eta)**2 + self.eta
        return loss, total_loss.sum()


def train_fun(model,criterion,optimizer,train_loader,val_loader,
              gradient_norm=1.0,scheduler=None,iteration=0,reg_lambda=0.0):
    # if init is not None:
    #     if iteration < 10:
    #         for param in model.initial_layer.parameters():
    #             param.requires_grad=False
    #         for param in model.trunk_layers.parameters():
    #             param.requires_grad=False
    #
    #     else:
    #         for param in model.parameters():
    #             param.requires_grad=True


    model.train()
    train_loss = 0
    total_train_loss=0
    train_accuracy1=0
    train_accuracy2=0
    train_accuracy3=0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
        logits=model(batch_x)
        losses,loss = criterion(logits, batch_y)
        weight_norm=torch.stack([torch.norm(v,1)for n,v in model.named_parameters() if "weight" in n]).sum()
        # l1_penalty = 0.
        # for output in output_hook:
        #     l1_penalty += torch.norm(output, 1)
        total_loss=loss+ reg_lambda * weight_norm
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm,)
        optimizer.step()
        train_loss += loss.item() / len(train_loader)
        total_train_loss += total_loss.item() / len(train_loader)
        train_pred = torch.stack([logit.argmax(dim=1) for logit in logits], axis=1)
        accuracy1, accuracy2, accuracy3 = torch.mean((train_pred == batch_y) * 1.0, dim=0).numpy()
        train_accuracy1 += accuracy1 / len(train_loader)
        train_accuracy2 += accuracy2 / len(train_loader)
        train_accuracy3 += accuracy3 / len(train_loader)
        # output_hook.clear()
    train_accuracy=np.mean([train_accuracy1,train_accuracy2,train_accuracy3])

    model.eval()
    val_loss = 0
    total_val_loss=0
    val_accuracy1=0
    val_accuracy2 = 0
    val_accuracy3 = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device,dtype=torch.float), batch_y.to(device,dtype=torch.int64)
            logits=model(batch_x)
            losses,loss = criterion(logits, batch_y)
            weight_norm = torch.stack([torch.norm(v, 1) for n, v in model.named_parameters() if "weight" in n]).sum()
            # z = torch.sum(torch.stack(encoder_hook.outputs, dim=0), dim=0)
            # l1_penalty = torch.norm(z, 1, 1).mean()
            # total_loss = loss - lambda_sparsity * M_loss+l1_lambda*l1_penalty
            # l1_penalty = 0.
            # for output in output_hook:
            #     l1_penalty += torch.norm(output, 1)
            total_loss = loss + reg_lambda * weight_norm
            val_loss += loss.item() / len(val_loader)
            total_val_loss += total_loss.item()/len(val_loader)
            val_pred=torch.stack([logit.argmax(dim=1) for logit in logits],axis=1)
            accuracy1,accuracy2,accuracy3=torch.mean((val_pred==batch_y)*1.0,dim=0).numpy()
            val_accuracy1+=accuracy1/len(val_loader)
            val_accuracy2 += accuracy2 / len(val_loader)
            val_accuracy3 += accuracy3 / len(val_loader)
    val_accuracy=np.mean([val_accuracy1,val_accuracy2,val_accuracy3])


            # output_hook.clear()
    if scheduler: scheduler.step()


    return train_loss,total_train_loss, val_loss,total_val_loss, train_accuracy,val_accuracy


kfold_results=[]
kfold_metrics=[]
best_configs=[]
folds=KFold(n_splits=5, random_state=123, shuffle=True)
for i,(split_train,split_test) in enumerate(folds.split(X,y)):
    X_train,y_train=X[split_train,:],y[split_train,:]
    X_valid, y_valid = X[split_test, :], y[split_test,:]

    scl=StandardScaler()
    X_train=scl.fit_transform(X_train)
    X_valid=scl.transform(X_valid)

    epochs=100


    class Trainer(tune.Trainable):
        def setup(self, config):
            self.model = get_model()
            self.optimizer = get_optimizer(config, self.model)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
            self.criterion = MultiTaskLoss(device=device)
            self.train_loader = get_train_loader(config, X_train, y_train)
            self.val_loader = get_val_loader(X_valid, y_valid)
            self.lambda_sparsity = config['l1_sparsity']
            self.gradient_norm = config['gradient_norm']

        def step(self):
            train_loss,total_train_loss, val_loss,total_val_loss, train_accuracy,val_accuracy = train_fun(self.model, self.criterion,
                                                                                    self.optimizer,
                                                                                    train_loader=self.train_loader,
                                                                                    val_loader=self.val_loader,
                                                                                    gradient_norm=self.gradient_norm,
                                                                                    scheduler=self.scheduler,
                                                                                    iteration=self.iteration,
                                                                                    reg_lambda=self.lambda_sparsity)
            return {'train_loss': train_loss, 'loss': val_loss, 'train_accuracy': train_accuracy,
                    'total_train_loss': total_train_loss, 'total_val_loss': total_val_loss,'val_accuracy':val_accuracy}

        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save((self.model.state_dict(), self.optimizer.state_dict()), checkpoint_path)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            model_state, optimizer_state = torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

        def reset_config(self, new_config):
            self.optimizer = get_optimizer(config, self.model)
            self.train_loader = get_train_loader(config, X_train, y_train)
            # self.lambda_sparsity = config['lambda_sparsity']
            self.gradient_norm = config['gradient_norm']
            self.config = new_config
            return True

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=50,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "val_accuracy","val_loss", "training_iteration"])
    result = tune.run(
        Trainer,
        checkpoint_at_end=True,
        # keep_checkpoints_num=2,
        # checkpoint_score_attr='min-loss',
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=configs,
        local_dir=os.path.join(log_dir,experiment),
        num_samples=100,
        name=f'fold{i}',
        resume=False,
        scheduler=scheduler,
        # stop={'training_iteration':500},
        progress_reporter=reporter,
        reuse_actors=False,
        raise_on_failed_trial=False)

    metric='loss'; mode='min';scope='last-5-avg'
    best_result=result.get_best_trial(metric,mode,scope=scope).last_result
    print(best_result)
    kfold_results.append(result)


    best_config=result.get_best_config(metric,mode,scope=scope)
    best_configs.append(best_config)

#stochastic weight averaging
X_test_scl=scl.transform(X_test)

for result in kfold_results:
    df=result.dataframe('loss','min')