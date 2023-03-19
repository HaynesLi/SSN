

import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

from collections import defaultdict
import os




# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/ssd/l5kit_dataset"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")
print(cfg)



model_path = "/tmp/planning.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = torch.load(model_path).to(device)
model = model.eval()
torch.set_grad_enabled(False)





# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
eval_dataloader= DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])


num_scenes_to_unroll = 63
num_simulation_steps = 160





# ==== DEFINE CLOSED-LOOP SIMULATION
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)



print(len(eval_zarr.scenes)//num_scenes_to_unroll)
print(len(eval_zarr.scenes))
scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
sim_outs = sim_loop.unroll(scenes_to_unroll)




metrics = [CollisionFrontMetric(),
           CollisionRearMetric(),
           CollisionSideMetric()]

validators = [RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
              RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
              RangeValidator("collision_side", CollisionSideMetric, max_value=0)]

intervention_validators = ["collision_front",
                           "collision_rear",
                           "collision_side"]

cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                                   validators=validators,
                                                   composite_metrics=[],
                                                   intervention_validators=intervention_validators))





cle_evaluator.evaluate(sim_outs)
validation_results = cle_evaluator.validation_results()
agg = ValidationCountingAggregator().aggregate(validation_results)
cle_evaluator.reset()





fields = ["metric", "value"]
table = PrettyTable(field_names=fields)

values = []
names = []

for metric_name in agg:
    table.add_row([metric_name, agg[metric_name].item()])
    values.append(agg[metric_name].item())
    names.append(metric_name)




plt.bar(np.arange(len(names)), values)
plt.xticks(np.arange(len(names)), names)
plt.show()





output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
for sim_out in sim_outs: # for each scene
    vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
    show(visualize(sim_out.scene_id, vis_in))




