
from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.rasterized.MyModel import MyModel
from l5kit.planning.rasterized import utils
#from l5kit.planning.rasterized.model import MyModel
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.planning.rasterized.earlystop import EarlyStopping,LRScheduler
import os




# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/ssd/l5kit_data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")




perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])
perturbation = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
val_zarr=ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)
val_dataset=EgoDataset(cfg, val_zarr, rasterizer)
print("train: ",train_dataset)
print("val: ",val_dataset)

perturbation.perturb_prob = perturb_prob



model = MyModel()



train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])

val_cfg=cfg["val_data_loader"]
val_dataloader =  DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"], 
                             num_workers=val_cfg["num_workers"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-6)#1e-7
lr_scheduler = LRScheduler(optimizer)

losses_train=[]
# print(train_dataset)

def fit(model, train_dataloader, optimizer):
    print('Training')
    tr_it = iter(train_dataloader)
    model.train()
    counter = 0
    train_running_loss = 0.0
    
    prog_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    for _ in prog_bar:
        counter += 1
        try:
            data=next(tr_it)
        except StopIteration:
            tr_it=iter(train_dataloader)
            data=next(iter)
        data = {k: v.to(device) for k, v in data.items()}
        result=model(data)
        loss=result["loss"]
        train_running_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        losses_train.append(loss.item())

        prog_bar.set_description(f"train_loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    train_loss=train_running_loss/counter

    return train_loss

losses_val=[]
def validate(model, val_dataloader,epoch):
    print('Validating')
    model.eval()
    val_it = iter(val_dataloader)
    counter = 0
    val_running_loss = 0.0
    prog_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    with torch.no_grad():
        for _ in prog_bar:
            try:
                data=next(val_it)
            except StopIteration:
                val_it=iter(val_dataloader)
                data=next(iter)
            counter+=1
            data = {k: v.to(device) for k, v in data.items()}

            result=model(data)
            loss=result["loss"]
            
            val_running_loss += loss.item()
            losses_val.append(loss.item())
            prog_bar.set_description(f"val_loss: {loss.item()} loss(avg): {np.mean(losses_val)}")
        
        val_loss=val_running_loss/counter

        torch.save(model,"/tmp/"+str(epoch)+"planning.pt")
        return val_loss


n=cfg["train_params"]["max_num_steps"]
for epoch in range(1000):
    print(f"Epoch {epoch+1} of {n}")
    train_epoch_loss = fit(
        model, train_dataloader,optimizer
    )
    val_epoch_loss = validate(
        model, val_dataloader,epoch
    )

    lr_scheduler(val_epoch_loss)

    early_stopping = EarlyStopping()
    early_stopping(val_epoch_loss)
    if early_stopping.early_stop:
        break

print("Finished")



to_save = torch.save(model,"/tmp/planning.pt")

plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.show()

