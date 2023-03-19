import warnings
from typing import Dict, List,Optional

import torch
import torch.nn as nn
from l5kit.planning.rasterized.SSN import SSNStem, Patch_Aggregate, SSNBlock


from l5kit.environment import models

class AverageMeter(val,n):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # def update(self,
    #     val: float,
    #     n: Optional[int] = 1
    # ) -> None:
    #     self.val = val
    #     self.sum += val * n
    #     self.count += n
    #     self.avg = self.sum / self.count
    #     return self.avg

    def update1(val,n):
        vv=val
        ss += vv*n
        count += n
        avg=ss/count
        return avg


class MyModel(nn.Module):
    def __init__(self,
                 in_channels=5,
                 stem_channel=32,
                 ssn_channel=[46, 92, 184, 368],
                 patch_channel=[46, 92, 184, 368],
                 block_layer=[2, 2, 10, 2],
                 R=3.6,
                 img_size=224,
                 num_class=36,
                 ):
        super(MyModel, self).__init__()

        size = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        self.stem = SSNStem(in_channels, stem_channel)
        self.patch1 = Patch_Aggregate(stem_channel, patch_channel[0])
        self.patch2 = Patch_Aggregate(patch_channel[0], patch_channel[1])
        self.patch3 = Patch_Aggregate(patch_channel[1], patch_channel[2])
        self.patch4 = Patch_Aggregate(patch_channel[2], patch_channel[3])

        stage1 = []
        for _ in range(block_layer[0]):
            ssn_layer = SSNBlock(
                img_size=size[0],
                stride=8,
                d_k=ssn_channel[0],
                d_v=ssn_channel[0],
                num_heads=1,
                R=R,
                in_channels=patch_channel[0]
            )
            stage1.append(ssn_layer)
        self.stage1 = nn.Sequential(*stage1)

        stage2 = []
        for _ in range(block_layer[1]):
            ssn_layer = SSNBlock(
                img_size=size[1],
                stride=4,
                d_k=ssn_channel[1] // 2,
                d_v=ssn_channel[1] // 2,
                num_heads=2,
                R=R,
                in_channels=patch_channel[1]
            )
            stage2.append(ssn_layer)
        self.stage2 = nn.Sequential(*stage2)

        stage3 = []
        for _ in range(block_layer[2]):
            ssn_layer = SSNBlock(
                img_size=size[2],
                stride=2,
                d_k=ssn_channel[2] // 4,
                d_v=ssn_channel[2] // 4,
                num_heads=4,
                R=R,
                in_channels=patch_channel[2]
            )
            stage3.append(ssn_layer)
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        for _ in range(block_layer[3]):
            ssn_layer = SSNBlock(
                img_size=size[3],
                stride=1,
                d_k=ssn_channel[3] // 8,
                d_v=ssn_channel[3] // 8,
                num_heads=8,
                R=R,
                in_channels=patch_channel[3]
            )
            stage4.append(ssn_layer)
        self.stage4 = nn.Sequential(*stage4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(ssn_channel[3], 1240),
            nn.ReLU(inplace=True),
        )

        # self.lstm1 = nn.LSTM(2048, 1024)
        # self.lstm2 = nn.LSTM(1024, 512)

        self.classifier=nn.Linear(1240, num_class)

        # self.model = MyModel()

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        x=data_batch["image"]
        print("xtype: ",x.type())

        # [batch_size, num_steps * 2]

        x=self.stem(x)
        # print("x: ",x)

        x=self.patch1(x)

        x=self.stage1(x)

        x=self.patch2(x)

        x=self.stage2(x)

        x=self.patch3(x)
        x=self.stage3(x)

        x=self.patch4(x)
        x=self.stage4(x)

        x=self.avg_pool(x)
        x=torch.flatten(x, 1)


        x=self.fc(x)
        # print(x.shape)

        # x=self.lstm1(x)
        # x=self.lstm2(x)

        outputs=self.classifier(x)
        # print("x: ", outputs)
        # outputs = MyModel(image_batch)

        batch_size=len(data_batch["image"])
        # return outputs

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_steps * 2]
            targets=(torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights=(data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            # train_loss =AverageMeter()
            loss = self.criterion(outputs, targets)
            # loss=torch.mean(self.criterion(outputs, targets) * target_weights)


            # train_loss.update(loss.item(), 64)

            train_dict={"loss": loss}
            return train_dict
        else:
            predicted=outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions=predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws=predicted[:, :, 2:3]
            eval_dict={"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict
