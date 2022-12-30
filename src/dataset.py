"""
create train or eval dataset.
"""
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication.management import init, get_rank, get_group_size
import random
import os
import time
import multiprocessing as mp
import numpy as np
import cv2


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized

def get_train_list(data_dir):
    g = os.walk(data_dir)
    path_list = []
    for path, dir_list, file_list in g:
        if path != data_dir:
            path_list.append(path)
    path_list.sort(key=lambda x: int(x[-8:-1]))
    label = 1
    path_label = []
    for out in path_list:
        z = os.walk(out)
        for path, dir_list, file_list in z:
            for i in file_list:
                img_path = i.split("_")[0] + "/" + str(i) + " " + str(label)
                path_label.append(img_path)
        label = label + 1
    return path_label


def get_val_list(data_dir):
    g = os.walk(data_dir)
    path_list = []
    for path, dir_list, file_list in g:
        if path != data_dir:
            path_list.append(path)
    path_list.sort(key=lambda x: int(x[-8:-1]))

    label = 1
    path_label = []
    for out in path_list:
        z = os.walk(out)
        for path, dir_list, file_list in z:
            for i in file_list:
                img_path = out.split("/")[-1] + "/" + str(i) + " " + str(label)
                path_label.append(img_path)
        label = label + 1
    return path_label



class GetDatasetGenerator_eval():
    """ GetDatasetGenerator_eval"""
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        train_image_list = []
        TRAIN_LISTS = get_val_list(data_dir)
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            train_image_list.append((os.path.join(self.DATA_DIR, path), label))
        f = open('./iamgenet_val.txt', 'a')
        for i in train_image_list:
            print(i)
            f.write(str(i)+'\n')
        self.__data = [i[0] for i in train_image_list]
        self.__label = [i[1] for i in train_image_list]
    def __getitem__(self, index):
        self.__img = cv2.imread(self.__data[index])
        self.__img = resize_short(self.__img, 224)
        item = (self.__img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)



class GetDatasetGenerator_softmax():
    """ GetDatasetGenerator_softmax """
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        train_image_list = []
        TRAIN_LISTS = get_train_list(data_dir)
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            train_image_list.append((os.path.join(self.DATA_DIR, path), label))
        r = random.random
        random.seed(int(time.time()))
        random.shuffle(train_image_list, random=r)
        self.__data = [i[0] for i in train_image_list]
        self.__label = [i[1] for i in train_image_list]
    def __getitem__(self, index):
        self.__img = cv2.imread(self.__data[index])
        item = (self.__img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)

class GetDatasetGenerator_triplet():
    """ GetDatasetGenerator_triplet """
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        train_data = {}
        train_image_list_tiplet = []
        TRAIN_LISTS = get_train_list(data_dir)
        count = 0
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            path = items[0]
            label = int(items[1]) - 1
            if label not in train_data:
                train_data[label] = []
            train_data[label].append(path)
        #shuffle
        r = random.random
        random.seed(int(time.time()))
        #data generates
        labs = list(train_data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        total_count = len(TRAIN_LISTS)
        while True:
            random.shuffle(ind, random=r)
            ind_pos, ind_neg = ind[:2]
            lab_pos = labs[ind_pos]
            pos_data_list = train_data[lab_pos]
            data_ind = list(range(0, len(pos_data_list)))
            random.shuffle(data_ind, random=r)
            anchor_ind, pos_ind = data_ind[:2]
            lab_neg = labs[ind_neg]
            neg_data_list = train_data[lab_neg]
            neg_ind = random.randint(0, len(neg_data_list) - 1)
            anchor_path = self.DATA_DIR +"/" + pos_data_list[anchor_ind]
            train_image_list_tiplet.append((anchor_path, lab_pos))
            pos_path = self.DATA_DIR +"/" + pos_data_list[pos_ind]
            train_image_list_tiplet.append((pos_path, lab_pos))
            neg_path = self.DATA_DIR +"/" + neg_data_list[neg_ind]
            train_image_list_tiplet.append((neg_path, lab_neg))
            count += 3
            # print(count)
            if count >= total_count:
                break

        f = open('./train_image_list_tiplet.txt', 'a')
        for i in train_image_list_tiplet:
            f.write(str(i)+'\n')
        self.__data = [i[0] for i in train_image_list_tiplet]
        self.__label = [i[1] for i in train_image_list_tiplet]

    def __getitem__(self, index):
        img = cv2.imread(self.__data[index])
        item = (img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)

def create_dataset2(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
                    target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info(distribute)

    ds.config.set_prefetch_size(2)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.c_transforms.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans = [
            ds.vision.c_transforms.Decode(),
            ds.vision.c_transforms.Resize(256),
            ds.vision.c_transforms.CenterCrop(eval_image_size)
        ]
    trans_norm = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]

    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.int32)
    if device_num == 1:
        trans_work_num = 24
    else:
        trans_work_num = 12
    data_set = data_set.map(operations=trans, input_columns="image")
    data_set = data_set.map(operations=trans_norm, input_columns="image")
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
    
def create_dataset_pynative(dataset_path, do_train, batch_size=32, train_image_size=224,
                            eval_image_size=224, target="Ascend", distribute=False, enable_cache=False,
                            cache_session_id=None):
    """
    create a train or eval imagenet2012 dataset for resnet50 benchmark

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info(distribute)

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path,  shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.c_transforms.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
            ds.vision.c_transforms.Normalize(mean=mean, std=std),
            ds.vision.c_transforms.HWC2CHW()
        ]
    else:
        trans = [
            ds.vision.c_transforms.Decode(),
            ds.vision.c_transforms.Resize(256),
            ds.vision.c_transforms.CenterCrop(eval_image_size),
            ds.vision.c_transforms.Normalize(mean=mean, std=std),
            ds.vision.c_transforms.HWC2CHW()
        ]

    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=trans, input_columns="image")
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def create_dataset_triplet(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
                    target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info(distribute)
    ds.config.set_prefetch_size(2)
    ds.config.set_enable_shared_mem(False)
    if device_num == 1:

        dataset_generator = GetDatasetGenerator_triplet(dataset_path)
        data_set = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
    else:
        dataset_generator = GetDatasetGenerator_triplet(dataset_path)
        data_set = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True,
                                        num_shards=device_num, shard_id=rank_id, python_multiprocessing=False)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.c_transforms.RandomResizedCrop(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans = [
            ds.vision.c_transforms.Resize(256),
            ds.vision.c_transforms.CenterCrop(eval_image_size)
        ]
    trans_norm = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]

    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.int32)
    if device_num == 1:
        trans_work_num = 24
    else:
        trans_work_num = 12
    data_set = data_set.map(operations=trans, input_columns="image")
    data_set = data_set.map(operations=trans_norm, input_columns="image")
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set

def create_datasetour_eval(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
                    target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info(distribute)
    ds.config.set_prefetch_size(2)
    ds.config.set_enable_shared_mem(False)
    if device_num == 1:
        dataset_generator = GetDatasetGenerator_eval(dataset_path)
        data_set = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
    else:
        dataset_generator = GetDatasetGenerator_eval(dataset_path)
        data_set = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)


    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.c_transforms.RandomResizedCrop(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans = [
            ds.vision.c_transforms.Resize(256),
            ds.vision.c_transforms.CenterCrop(eval_image_size)
        ]
    trans_norm = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]

    type_cast_op = ds.transforms.c_transforms.TypeCast(ms.int32)
    if device_num == 1:
        trans_work_num = 24
    else:
        trans_work_num = 12
    data_set = data_set.map(operations=trans, input_columns="image")
    data_set = data_set.map(operations=trans_norm, input_columns="image")
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def _get_rank_info(distribute):
    """
    get rank size and rank id
    """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id