# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""eval resnet."""
import os
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from mindspore import context
import moxing as mox

workroot = '/home/work/user-job-dir'
ms.set_seed(1)

if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152"):
    if config.net_name == "resnet18":
        from src.resnet import resnet18 as resnet
    elif config.net_name == "resnet34":
        from src.resnet import resnet34 as resnet
    elif config.net_name == "resnet50":
        from src.eval_resnet import resnet50 as resnet
    else:
        from src.resnet import resnet152 as resnet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.dataset import create_dataset2 as create_dataset
elif config.net_name == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.dataset import create_dataset4 as create_dataset

    
if __name__ == '__main__':

    #初始化数据存放目录
    data_url = workroot + '/data'
    if not os.path.exists(data_url):
        os.mkdir(data_url)

    #初始化模型存放目录
    train_url = workroot + '/model/'
    if not os.path.exists(train_url):
            os.mkdir(train_url)
            
    local_train_url = train_url
    print(os.path.exists(train_url))

    #将数据集从local拷贝到推理镜像中：
    local_data_url = config.data_url
    print(os.path.exists(local_data_url))
    config.data_url = '/home/work/user-job-dir/data/'
    try:    
        mox.file.copy_parallel(local_data_url, config.data_url)
        print("Successfully Download {} to {}".format(local_data_url,config.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(local_data_url, config.data_url) + str(e))



    #将模型文件从local拷贝到推理镜像中：
    local_ckpt_url = config.ckpt_url
    config.ckpt_url = '/home/work/user-job-dir/checkpoint.ckpt'
    try:
        mox.file.copy(local_ckpt_url, config.ckpt_url)
        print("Successfully Download {} to {}".format(local_ckpt_url,
                                                    config.ckpt_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            local_ckpt_url, config.ckpt_url) + str(e))

    config.checkpoint_file_path = config.ckpt_url
    
    # init context
    ms.context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
    device_id = int(os.getenv('DEVICE_ID'))
    ms.context.set_context(device_id=device_id)
        
    # create dataset
    DATA_DIR = '/home/work/user-job-dir/data/imagenet/val'
    dataset = create_dataset(dataset_path=DATA_DIR, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size,
                             target='Ascend')

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = ms.load_checkpoint(config.checkpoint_file_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", config.checkpoint_file_path)