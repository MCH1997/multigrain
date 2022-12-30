# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
config.
"""
from easydict import EasyDict as ed

#sop trpletloss
config1 = ed({
    "optimizer": "Momentum",
    "net_name": "resnet50",
    "class_num": 1001,
    "batch_size": 240,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "epoch_size": 120,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./triplet/",
    "warmup_epochs": 2,
    "lr_decay_mode": "linear",
    "lr_init": 0,
    "lr_end": 0.0,
    "lr_max": 0.8,
    "lr_decay": 4.0839,
    "lr_end_epoch": 53,
    "train_image_size": 224,
    "eval_image_size": 224,
    "run_distribute": True
})