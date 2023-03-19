
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
# plot same example with and without perturbation
# for perturbation_value in [1, 0]:
#     perturbation.perturb_prob = perturbation_value

#     data_ego = train_dataset[0]
#     im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
#     target_positions = transform_points(data_ego["target_positions"], data_ego["raster_from_agent"])
#     draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
#     plt.imshow(im_ego)
#     plt.axis('off')
#     plt.show()

# before leaving, ensure perturb_prob is correct
perturbation.perturb_prob = perturb_prob



model = MyModel()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# if torch.cuda.device_count()>0:
#     model=nn.DataParallel(model)

# model = model.to(device)

# print(model)


# model = RasterizedPlanningModel(
#         model_arch="resnet50",
#         num_input_channels=rasterizer.num_channels(),
#         num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
#         weights_scaling= [1., 1., 1.],
#         criterion=nn.MSELoss(reduction="none")
#         )



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

#written and saved in train.py
# lists to store per-epoch loss and accuracy values
# train_loss = []
# val_loss = []
# start = time.time()
n=cfg["train_params"]["max_num_steps"]
for epoch in range(10):
    print(f"Epoch {epoch+1} of {n}")
    train_epoch_loss = fit(
        model, train_dataloader,optimizer
    )
    val_epoch_loss = validate(
        model, val_dataloader,epoch
    )
    # train_loss.append(train_epoch_loss)
    # train_accuracy.append(train_epoch_accuracy)
    # val_loss.append(val_epoch_loss)
    # val_accuracy.append(val_epoch_accuracy)
    # if args['lr_scheduler']:
    lr_scheduler(val_epoch_loss)
    # if args['early_stopping']:
    early_stopping = EarlyStopping()
    early_stopping(val_epoch_loss)
    if early_stopping.early_stop:
        break
    # print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    # print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
# end = time.time()
print("Finished")

# tr_it = iter(train_dataloader)
# progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
# losses_train = []
# model.train()
# torch.set_grad_enabled(True)

# for _ in progress_bar:
#     try:
#         data = next(tr_it)
#     except StopIteration:
#         tr_it = iter(train_dataloader)
#         data = next(tr_it)
#     # Forward pass
#     data = {k: v.to(device) for k, v in data.items()}
#     # print("data: ",data)
#     result = model(data)
#     loss = result["loss"]
#     print("loss: ",loss)
#     # loss=result["train_acc1"]
#     # train_loss=AverageMeter.update1(loss.item(),12)
#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     # train_loss.backward()
#     optimizer.step()

#     losses_train.append(loss.item())
#     # losses_train.append(train_loss)
#     progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
#     # progress_bar.set_description(f"loss: {train_loss} loss(avg): {np.mean(losses_train)}")


to_save = torch.save(model,"/tmp/planning.pt")

plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.show()


# save_path = "/tmp" #当前目录下
# early_stopping = EarlyStopping(save_path)

# to_save = torch.save(model,"/tmp/planning.pt")

# path_to_save = f"{gettempdir()}/planning_model.pt"
# to_save.save(path_to_save)
# print(f"MODEL STORED at {path_to_save}")

