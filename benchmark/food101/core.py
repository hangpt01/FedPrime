import functools
import time
from .dataset import FOOD101Dataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import wandb
from tqdm import tqdm
import warnings
import collections
import torch
import torch.nn.functional as F
from datetime import datetime
warnings.filterwarnings('ignore')
    
class TaskPipe(IDXTaskPipe):
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        
        _config = {
            'missing_ratio':
                {'test': feddata['datasrc']['missing_ratio']['train'],
                'train': feddata['datasrc']['missing_ratio']['test']},
            'missing_table_root': feddata['datasrc']['missing_table_root'],
            'missing_type':
                {'train': feddata['datasrc']['missing_type']['train'],
                'test': feddata['datasrc']['missing_type']['test']},
            'both_ratio': feddata['datasrc']['both_ratio'],
            'simulate_missing': False
        }
        
        origin_class = getattr(importlib.import_module(class_path), class_name)
        # import pdb; pdb.set_trace()
        data_dir = "./benchmark/RAW_DATA/FOOD101/generate_arrows"
        transform_keys = ['pixelbert']
        split="train"
        image_size = 384
        # TO_DELETE
        max_text_len = 40
        draw_false_image = 0
        draw_false_text = 0
        image_only = False
        missing_info = {
                'ratio' : _config["missing_ratio"],
                'type' : _config["missing_type"],
                'both_ratio' : _config["both_ratio"],
                'missing_table_root': _config["missing_table_root"],
                'simulate_missing' : _config["simulate_missing"]
            }        
        # import pdb; pdb.set_trace()
        # origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        # origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        # import pdb; pdb.set_trace()
        collator = DataCollatorForLanguageModeling

        origin_train_data = FOOD101Dataset(data_dir, transform_keys, split='train', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
        origin_train_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        # origin_train_data.mlm_collator = collator(tokenizer=origin_train_data.tokenizer, mlm=True, mlm_probability=0.15)
        # origin_train_data.collate = functools.partial(origin_train_data.collate, mlm_collator=origin_train_data.mlm_collator)

        origin_test_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
        origin_test_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        # origin_test_data.mlm_collator = collator(tokenizer=origin_test_data.tokenizer, mlm=True, mlm_probability=0.15)
        # origin_test_data.collate = functools.partial(origin_test_data.collate, mlm_collator=origin_test_data.mlm_collator)
        # import pdb; pdb.set_trace()
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])

        # import pdb; pdb.set_trace()
        # other test data
        other_test_datas = []
        other_test_names = []
        
        if missing_info['type']['train'] != "image":
            missing_image_config = {
            'ratio':
                {'test': 0.7,
                'train': 0.7},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
            'type':
                {'test': 'image',
                'train': 'both'},
            'both_ratio': 0,
            'simulate_missing': False
            }
            origin_test_miss_image_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                    image_size=image_size,
                                    max_text_len=max_text_len,
                                    draw_false_image=draw_false_image,
                                    draw_false_text=draw_false_text,
                                    image_only=image_only,
                                    missing_info=missing_image_config)
            origin_test_miss_image_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
            test_miss_image_data = cls.TaskDataset(origin_test_miss_image_data, [_ for _ in range(len(origin_test_miss_image_data))])
            other_test_datas.append(test_miss_image_data)
            other_test_names.append("miss_image")

        # import pdb; pdb.set_trace()
        
        if missing_info['type']['train'] != "text":
            missing_text_config = {
            'ratio':
                {'test': 0.7,
                'train': 0.7},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
            'type':
                {'test': 'text',
                'train': 'both'},
            'both_ratio': 0,
            'simulate_missing': False
            }
            origin_test_miss_text_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                    image_size=image_size,
                                    max_text_len=max_text_len,
                                    draw_false_image=draw_false_image,
                                    draw_false_text=draw_false_text,
                                    image_only=image_only,
                                    missing_info=missing_text_config)
            origin_test_miss_text_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
            test_miss_text_data = cls.TaskDataset(origin_test_miss_text_data, [_ for _ in range(len(origin_test_miss_text_data))])
            other_test_datas.append(test_miss_text_data)
            other_test_names.append("miss_text")

        if missing_info['type']['train'] != "both":
            missing_both_config = {
            'ratio':
                {'test': 0.7,
                'train': 0.7},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
            'type':
                {'test': 'both',
                'train': 'both'},
            'both_ratio': 0.5,
            'simulate_missing': False
            }
            origin_test_miss_both_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                    image_size=image_size,
                                    max_text_len=max_text_len,
                                    draw_false_image=draw_false_image,
                                    draw_false_text=draw_false_text,
                                    image_only=image_only,
                                    missing_info=missing_both_config)
            origin_test_miss_both_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
            test_miss_both_data = cls.TaskDataset(origin_test_miss_both_data, [_ for _ in range(len(origin_test_miss_both_data))])
            other_test_datas.append(test_miss_both_data)
            other_test_names.append("miss_both")
    

        full_modal_config = {
        'ratio':
            {'test': 0,
            'train': 0.7},
        'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
        'type':
            {'test': 'both',
            'train': 'both'},
        'both_ratio': 0,
        'simulate_missing': False
        }
        origin_test_full_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=full_modal_config)
        origin_test_full_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        test_full_data = cls.TaskDataset(origin_test_full_data, [_ for _ in range(len(origin_test_full_data))])
        other_test_datas.append(test_full_data)
        other_test_names.append("full_modal")


        image_only_config = {
        'ratio':
            {'test': 1,
            'train': 0.7},
        'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
        'type':
            {'test': 'text',
            'train': 'both'},
        'both_ratio': 0,
        'simulate_missing': False
        }
        origin_test_image_only_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=image_only_config)
        origin_test_image_only_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        test_image_only_data = cls.TaskDataset(origin_test_image_only_data, [_ for _ in range(len(origin_test_image_only_data))])
        other_test_datas.append(test_image_only_data)
        other_test_names.append("image_only")


        text_only_config = {
        'ratio':
            {'test': 1,
            'train': 0.7},
        'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
        'type':
            {'test': 'image',
            'train': 'both'},
        'both_ratio': 0,
        'simulate_missing': False
        }
        origin_test_text_only_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=text_only_config)
        origin_test_text_only_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        test_text_only_data = cls.TaskDataset(origin_test_text_only_data, [_ for _ in range(len(origin_test_text_only_data))])
        other_test_datas.append(test_text_only_data)
        other_test_names.append("text_only")


        train_datas = []
        # valid_datas = []
        # modalities_list = []
        for name in feddata['client_names']:
            train_data = feddata[name]['dtrain']    # sample idx
            train_datas.append(cls.TaskDataset(origin_train_data, train_data))
        
        other_test_datas_combined = (other_test_datas, other_test_names)
        test_data = (test_data, other_test_datas_combined)
        return train_datas, test_data, feddata['client_names']

def collate(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert (
            len(size) == 3
        ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, max_height, max_width)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask
            # import pdb; pdb.set_trace()
            
    return dict_batch


def save_task(generator):
    """
    Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
    The input 'generator' must have attributes:
        :taskpath: string. the path of storing
        :train_data: the training dataset which is a dict {'x':..., 'y':...}
        :test_data: the testing dataset which is a dict {'x':..., 'y':...}
        :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
        :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
        :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
        :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                For example, for MNIST using this task pipe, the source_dict should be like:
                {'class_path': 'torchvision.datasets',
                    'class_name': 'MNIST',
                    'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                    'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                }
        :return:
    """
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    # import pdb; pdb.set_trace()
    for cid in range(len(generator.cnames)):
        # print(cid)
        # import pdb; pdb.set_trace()
        # if generator.specific_training_leads:
        #     feddata[generator.cnames[cid]] = {
        #         'modalities': generator.specific_training_leads[cid],
        #         'dtrain': generator.train_cidxs[cid],
        #         'dvalid': generator.valid_cidxs[cid]
        #     }
        # else:
        feddata[generator.cnames[cid]] = {
            'dtrain': generator.train_cidxs[cid]
        }
    # with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
    #     ujson.dump(feddata, outf)
    # return
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf, default=convert)
    return

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)        #cluster=class

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = np.random.randint(0, len(dataset), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}       # class:[indices]
    for idx in selected_indices:
        # import pdb; pdb.set_trace()
        label = dataset[idx]['label']
        group_id = label2cluster[int(label)]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_indices = [[] for _ in range(n_clients)]        
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster
    
    for cluster_id in range(n_clusters):
            weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
            clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    
    clients_counts = np.cumsum(clients_counts, axis=1)

    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            # clients_indices[client_id] += list(indices)
            clients_indices[client_id] += indices
            # import pdb; pdb.set_trace()

                    
    return clients_indices


def noniid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.labels)
    # import pdb; pdb.set_trace()
    clients_indices = by_labels_non_iid_split(generator.train_data, labels.shape[0], generator.num_clients, labels.shape[0], generator.skewness, frac=1, seed=generator.seed)
    # import pdb; pdb.set_trace()
    return clients_indices  

def iid_partition(generator):
    print(generator)
    # import pdb; pdb.set_trace()
    labels = np.unique(generator.train_data.labels)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.labels == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    # import pdb; pdb.set_trace()
    return local_datas

# def local_holdout(self, local_datas, shuffle=False):
#         """split each local dataset into train data and valid data according the rate."""
#         train_cidxs = []
#         valid_cidxs = []
#         for local_data in local_datas:
#             if shuffle:
#                 np.random.shuffle(local_data)
#             k = int(len(local_data) * (1-self.local_holdout_rate))
#             train_cidxs.append(local_data[:k])
#             valid_cidxs.append(local_data[k:])
#         return train_cidxs, valid_cidxs

# def get_food101_loader(dataset, batch_size=40, shuffle=True, num_workers=8, vocab_size=30522):
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate)
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, missing=False, missing_ratio_train=0.7, missing_ratio_test=0.7, missing_type_train='both', missing_type_test='both', both_ratio=0.5, max_text_len=40):
        super(TaskGen, self).__init__(benchmark='food101',
                                      dist_id=dist_id, 
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/FOOD101',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed)
        if self.dist_id==0:
            self.partition = iid_partition
        else: 
            self.partition = noniid_partition
        
        self.num_classes=8
        self.save_task=save_task
        self.visualize=self.visualize_by_class
        # import pdb; pdb.set_trace()
        # self.rawdata_path = os.path.join(self.rawdata_path, str(self.num_classes)+'_classes')
        self.source_dict = {
            'class_path': 'benchmark.food101.dataset',
            'class_name': 'FOOD101Dataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train': 'True',
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train':' False'
            },
            'missing_ratio': {
                'train': missing_ratio_train,
                'test': missing_ratio_test 
            },
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/',
            'missing_type': {
                'train': missing_type_train,
                'test': missing_type_test 
            },
            'both_ratio': both_ratio,
            'simulate_missing': False
        }
        self.data_dir = os.path.join(self.rawdata_path, 'generate_arrows')
        _config = {
            'missing_ratio':
                {'train': missing_ratio_train,
                'test': missing_ratio_test},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/',
            'missing_type':
                {'train': missing_type_train,
                'test': missing_type_test},
            'both_ratio': both_ratio,
            'simulate_missing': False
        }
        self.missing_info = {
                'ratio' : _config["missing_ratio"],
                'type' : _config["missing_type"],
                'both_ratio' : _config["both_ratio"],
                'missing_table_root': _config["missing_table_root"],
                'simulate_missing' : _config["simulate_missing"]
            }
        self.transform_keys = ['pixelbert']
        self.image_size = 384
        self.max_text_len = max_text_len
        self.draw_false_image = 0
        self.draw_false_text = 0
        self.image_only = False

        self.missing=missing
        # self.local_holdout_rate = 0.2
        # self.specific_training_leads = None
        
        if self.missing:
            self.taskname = self.taskname + '_' + 'missing_ratio_' + str(missing_ratio_train) + '_' + str(missing_ratio_test)  \
                                          + '_' + 'missing_type_' + str(missing_type_train) + '_' + str(missing_type_test) \
                                          + '_' + 'both_ratio_' + str(both_ratio)
        
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def load_data(self):
            collator = DataCollatorForLanguageModeling

            self.train_data = FOOD101Dataset(self.data_dir, self.transform_keys, split='train', 
                                    image_size=self.image_size,
                                    max_text_len=self.max_text_len,
                                    draw_false_image=self.draw_false_image,
                                    draw_false_text=self.draw_false_text,
                                    image_only=self.image_only,
                                    missing_info=self.missing_info)
            self.train_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
            self.train_data.mlm_collator = collator(tokenizer=self.train_data.tokenizer, mlm=True, mlm_probability=0.15)
            self.train_data.collate = functools.partial(self.train_data.collate, mlm_collator=self.train_data.mlm_collator)
            # import pdb; pdb.set_trace()
            self.test_data = FOOD101Dataset(self.data_dir, self.transform_keys, split='test', 
                                    image_size=self.image_size,
                                    max_text_len=self.max_text_len,
                                    draw_false_image=self.draw_false_image,
                                    draw_false_text=self.draw_false_text,
                                    image_only=self.image_only,
                                    missing_info=self.missing_info)
            self.test_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
            self.test_data.mlm_collator = collator(tokenizer=self.test_data.tokenizer, mlm=True, mlm_probability=0.15)
            self.test_data.collate = functools.partial(self.test_data.collate, mlm_collator=self.test_data.mlm_collator)
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads=2
        self.DataLoader = DataLoader
        
    def get_data_loader(self, dataset, batch_size=40, shuffle=True, num_workers=8, vocab_size=30522):
        # import pdb; pdb.set_trace()
        collator = DataCollatorForLanguageModeling
        tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        mlm_collator = collator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, collate_fn=functools.partial(collate, mlm_collator=mlm_collator))
        # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def data_to_device(self, data):
        # for k, v in data.items():
        #     print(k,len(v))
        batch = data
        # import pdb; pdb.set_trace()
        batch['image'][0] = batch['image'][0].to(self.device)
        for key in ['text_ids', 'text_labels', 'text_ids_mlm', 'text_labels_mlm', 'text_masks']:
            new_ls = []
            for tensor in data[key]:
                new_ls.append(tensor.to(self.device)) 
            batch[key] = torch.stack(new_ls)
        # batch = {k:v.to(self.device) for k,v in data.items()}
        # import pdb; pdb.set_trace()
        return batch
        
    def train_one_step(self, model, transformer, text_embeddings, data, client_id=None, current_round=None):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        # import pdb; pdb.set_trace()

        batch = self.data_to_device(data)
        # import pdb; pdb.set_trace()
        model.to(self.device) # y.device
        transformer.to(self.device)
        text_embeddings.to(self.device)
        # print(tdata[0])
        loss = model(transformer, text_embeddings, batch, "train", client_id, current_round)
        # backbone.to('cpu')
        # print(loss)
        return {'loss': loss}
    
    
    @torch.no_grad()
    def test(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        # print("\nIn core", batch_size)
        if batch_size==-1:batch_size=len(dataset)
        # print("Creating dataloader", "batch_size {} shuffle {} num_workers {}".format(batch_size, False, num_workers))
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        total_loss = 0.0
        labels = list()
        predicts = list()
        model.to(self.device) # y.device
        transformer.to(self.device)
        text_embeddings.to(self.device)
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # import pdb; pdb.set_trace()
            loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
            total_loss += loss.item()
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / (batch_id+1),
            'acc': accuracy
        }
    

    @torch.no_grad()
    def test_specific_data(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0, client_id=-1, option=None, current_round=-1):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        model.to(self.device) # y.device
        transformer.to(self.device)
        text_embeddings.to(self.device)
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # import pdb; pdb.set_trace()
            loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
            total_loss += loss.item()
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        # print("Client {}\n".format(client_id+1), labels, predicts)
        # if current_round % 25 == 0:
        #     confusion_matrix_save_path = 'fedtask/' + option['task'] + '/plot_confusion_matrix/' + option['model']
        #     if not os.path.exists(confusion_matrix_save_path):
        #         os.makedirs(confusion_matrix_save_path)
        #     confusion_matrix_save_file = confusion_matrix_save_path + '/client_{}_confusion_matrix_round'.format(client_id+1) + str(current_round)
        #     list_class = list(range(1,9))
        #     if option['wandb']:
        #         plot_confusion_matrix(labels, predicts, 'client_{}'.format(client_id+1), current_round, confusion_matrix_save_file, list_class)
        return {
            'loss': total_loss / (batch_id+1),
            'acc': accuracy
        }

    @torch.no_grad()
    def evaluate(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Evaluate metric on client model
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1: batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        labels = list()
        predicts = list()
        model.to(self.device) # y.device
        transformer.to(self.device)
        text_embeddings.to(self.device)
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            loss, loss_, outputs = model(transformer, text_embeddings, batch_data)
            loss = torch.tensor(loss).to(loss[0].device)
            if batch_id==0:
                total_loss = loss
            else:
                total_loss = loss + total_loss
            # TO_DELETE
            # if batch_id==0:
            #     break
        loss_eval = loss / (batch_id + 1) 
        # import pdb; pdb.set_trace()
        loss_eval = [loss for loss in loss_eval]
        
        return loss_eval

    
    @torch.no_grad()
    def server_test(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0, option=None, current_round=-1):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST-----------------------")
        # print("Starting server test", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        # for test_combi_index in range(len(2)):
        total_loss = 0.0
        labels = list()
        predicts = list()   
        # print("     Starting batch test", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # if batch_id==0:
            #     print("          Starting 1 inference", datetime.now())
            loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data, flag="test", current_round=current_round)
            # if batch_id==0:
            #     print("          End 1 inference", datetime.now())
            total_loss += loss.item() * len(batch_data['label'])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # if batch_id==0:
            #     print("     End each batch test", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test", datetime.now())
        return result
        
    
    @torch.no_grad()
    def server_other_test(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST-----------------------")
        # print("Starting server_other_test", datetime.now())
        model.eval()
        # import pdb; pdb.set_trace()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()   
            # print("     Starting batch test", datetime.now())
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # print("          Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("          End 1 inference", datetime.now())
                total_loss += loss.item() * len(batch_data['label'])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
                # print("     End each batch test", datetime.now()) 
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End server_other_test", datetime.now())
        return result


    @torch.no_grad()
    def server_test_soft_voting(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST_SOFT_VOTING-----------------------")
        # print("Starting server test - Soft voting", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        total_loss = 0.0
        labels = list()
        predicts = list()   

        # print("     Starting batch test - Soft voting", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            avg_loss = 0
            avg_probabilities = list()
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # print("          Starting each ensembled model - Soft voting", datetime.now())
            
            for glo_pool in model.client_global_pools:
                model.global_pool = glo_pool
                # print("               Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("               End 1 inference", datetime.now())
                probabilities = F.softmax(outputs, dim=1)
                # if avg_probabilities.device != probabilities.device:
                #     avg_probabilities = avg_probabilities.to(probabilities.device)
                # import pdb; pdb.set_trace()
                avg_probabilities.append(probabilities)
                avg_loss += loss
                # print("          End each ensembled model - Soft voting", datetime.now())
                
            # print("Average prob", avg_probabilities)
            # import pdb; pdb.set_trace()
            avg_probabilities = torch.mean(torch.stack(avg_probabilities), dim=0)
            # avg_probabilities /= len(model.client_local_prompts)
            avg_loss /= len(model.client_global_pools)

            predicted_classes = torch.argmax(avg_probabilities, dim=1)
            total_loss += avg_loss.item() * len(batch_data['label'])
            # import pdb; pdb.set_trace()
            predicts.extend(predicted_classes.cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
            # print("     End batch test - Soft voting", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test - Soft voting", datetime.now())

        # print(accuracy)
        # import pdb; pdb.set_trace()
        return result

    @torch.no_grad()
    def server_test_median_soft_voting(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST_SOFT_VOTING-----------------------")
        # print("Starting server test - Soft voting", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        total_loss = 0.0
        labels = list()
        predicts = list()   

        # print("     Starting batch test - Soft voting", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            avg_loss = 0
            avg_probabilities = list()
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # print("          Starting each ensembled model - Soft voting", datetime.now())
            
            for glo_pool in model.client_global_pools:
                model.global_pool = glo_pool
                # print("               Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("               End 1 inference", datetime.now())
                probabilities = F.softmax(outputs, dim=1)
                # if avg_probabilities.device != probabilities.device:
                #     avg_probabilities = avg_probabilities.to(probabilities.device)
                # import pdb; pdb.set_trace()
                avg_probabilities.append(probabilities)
                avg_loss += loss
                # print("          End each ensembled model - Soft voting", datetime.now())
                
            # print("Average prob", avg_probabilities)
            # import pdb; pdb.set_trace()
            median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
            # avg_probabilities /= len(model.client_local_prompts)
            avg_loss /= len(model.client_global_pools)

            predicted_classes = torch.argmax(median_probabilities, dim=1)
            total_loss += avg_loss.item() * len(batch_data['label'])
            # import pdb; pdb.set_trace()
            predicts.extend(predicted_classes.cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
            # print("     End batch test - Soft voting", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test - Soft voting", datetime.now())

        # print(accuracy)
        # import pdb; pdb.set_trace()
        return result


    @torch.no_grad()
    def server_test_kmeans_median_soft_voting(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST_SOFT_VOTING-----------------------")
        # print("Starting server test - Soft voting", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        total_loss = 0.0
        labels = list()
        predicts = list()   

        # Get representative pools using clustering based on prompt similarity
        representative_pools = get_representative_pools(model, n_clusters=3)

        # print("     Starting batch test - Soft voting", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            avg_loss = 0
            avg_probabilities = list()
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # print("          Starting each ensembled model - Soft voting", datetime.now())
            
            for glo_pool in representative_pools:
                model.global_pool = glo_pool
                # print("               Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("               End 1 inference", datetime.now())
                probabilities = F.softmax(outputs, dim=1)
                # if avg_probabilities.device != probabilities.device:
                #     avg_probabilities = avg_probabilities.to(probabilities.device)
                # import pdb; pdb.set_trace()
                avg_probabilities.append(probabilities)
                avg_loss += loss
                # print("          End each ensembled model - Soft voting", datetime.now())
                
            # print("Average prob", avg_probabilities)
            # import pdb; pdb.set_trace()
            median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
            # avg_probabilities /= len(model.client_local_prompts)
            avg_loss /= len(model.client_global_pools)

            predicted_classes = torch.argmax(median_probabilities, dim=1)
            total_loss += avg_loss.item() * len(batch_data['label'])
            # import pdb; pdb.set_trace()
            predicts.extend(predicted_classes.cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
            # print("     End batch test - Soft voting", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test - Soft voting", datetime.now())

        # print(accuracy)
        # import pdb; pdb.set_trace()
        return result
    

    @torch.no_grad()
    def server_test_agglo_median_soft_voting(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST_SOFT_VOTING-----------------------")
        # print("Starting server test - Soft voting", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        total_loss = 0.0
        labels = list()
        predicts = list()   

        # Get representative pools using clustering based on prompt similarity
        representative_pools = get_representative_pools_agglomerative(model, n_clusters=3)

        # print("     Starting batch test - Soft voting", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            avg_loss = 0
            avg_probabilities = list()
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # print("          Starting each ensembled model - Soft voting", datetime.now())
            
            for glo_pool in representative_pools:
                model.global_pool = glo_pool
                # print("               Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("               End 1 inference", datetime.now())
                probabilities = F.softmax(outputs, dim=1)
                # if avg_probabilities.device != probabilities.device:
                #     avg_probabilities = avg_probabilities.to(probabilities.device)
                # import pdb; pdb.set_trace()
                avg_probabilities.append(probabilities)
                avg_loss += loss
                # print("          End each ensembled model - Soft voting", datetime.now())
                
            # print("Average prob", avg_probabilities)
            # import pdb; pdb.set_trace()
            median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
            # avg_probabilities /= len(model.client_local_prompts)
            avg_loss /= len(model.client_global_pools)

            predicted_classes = torch.argmax(median_probabilities, dim=1)
            total_loss += avg_loss.item() * len(batch_data['label'])
            # import pdb; pdb.set_trace()
            predicts.extend(predicted_classes.cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
            # print("     End batch test - Soft voting", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test - Soft voting", datetime.now())

        # print(accuracy)
        # import pdb; pdb.set_trace()
        return result

    @torch.no_grad()
    def server_test_agglo_mean_soft_voting(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_TEST_SOFT_VOTING-----------------------")
        # print("Starting server test - Soft voting", datetime.now())
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        total_loss = 0.0
        labels = list()
        predicts = list()   

        # Get representative pools using clustering based on prompt similarity
        representative_pools = get_representative_pools_agglomerative(model, n_clusters=3)

        # print("     Starting batch test - Soft voting", datetime.now())
        for batch_id, batch_data in enumerate(data_loader):
            avg_loss = 0
            avg_probabilities = list()
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # print("          Starting each ensembled model - Soft voting", datetime.now())
            
            for glo_pool in representative_pools:
                model.global_pool = glo_pool
                # print("               Starting 1 inference", datetime.now())
                loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                # print("               End 1 inference", datetime.now())
                probabilities = F.softmax(outputs, dim=1)
                # if avg_probabilities.device != probabilities.device:
                #     avg_probabilities = avg_probabilities.to(probabilities.device)
                # import pdb; pdb.set_trace()
                avg_probabilities.append(probabilities)
                avg_loss += loss
                # print("          End each ensembled model - Soft voting", datetime.now())
                
            # print("Average prob", avg_probabilities)
            # import pdb; pdb.set_trace()
            avg_probabilities = torch.mean(torch.stack(avg_probabilities), dim=0)
            # avg_probabilities /= len(model.client_local_prompts)
            avg_loss /= len(model.client_global_pools)

            predicted_classes = torch.argmax(avg_probabilities, dim=1)
            total_loss += avg_loss.item() * len(batch_data['label'])
            # import pdb; pdb.set_trace()
            predicts.extend(predicted_classes.cpu().tolist())
            # TO_DELETE
            # if batch_id==0:
            #     break
            # print("     End batch test - Soft voting", datetime.now())

        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # print("End server test - Soft voting", datetime.now())

        # print(accuracy)
        # import pdb; pdb.set_trace()
        return result


    @torch.no_grad()
    def server_other_test_soft_voting(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST_SOFT_VOTING-----------------------")
        # print("Starting other server test - Soft voting", datetime.now())
        model.eval()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
        result = dict() 
        for i in range(len(datasets)):
            dataset = datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()
            # print("     Starting batch test - Soft voting", datetime.now())
            
            for batch_id, batch_data in enumerate(data_loader):
                avg_loss = 0
                avg_probabilities = list()
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # ls_prompt = [torch.sum(i) for i in model.client_local_prompts]
                # print("Sum prompts in testing", ls_prompt)
                # print("          Starting each ensembled model - Soft voting", datetime.now())
                for glo_pool in model.client_global_pools:
                    model.global_pool = glo_pool
                    # print(model.pool.prompt[model.pool.top_k_idx])
                    # print(torch.sum(local_prompt))
                    # import pdb; pdb.set_trace()
                    # print("               Starting 1 inference", datetime.now())
                    loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                    # print("               End 1 inference", datetime.now())
                    probabilities = F.softmax(outputs, dim=1)
                    # if avg_probabilities.device != probabilities.device:
                    #     avg_probabilities = avg_probabilities.to(probabilities.device)
                    avg_probabilities.append(probabilities)
                    avg_loss += loss
                    # print("          End each ensembled model - Soft voting", datetime.now())
                
                
                avg_probabilities = torch.mean(torch.stack(avg_probabilities), dim=0)
                avg_loss /= len(model.client_global_pools)

                predicted_classes = torch.argmax(avg_probabilities, dim=1)
                total_loss += avg_loss.item() * len(batch_data['label'])
                # import pdb; pdb.set_trace()
                predicts.extend(predicted_classes.cpu().tolist())
                # print("\tEnd batch test - Soft voting", datetime.now())
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End other server test - Soft voting", datetime.now())
            
        return result

    @torch.no_grad()
    def server_other_test_median_soft_voting(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST_SOFT_VOTING-----------------------")
        # print("Starting other server test - Soft voting", datetime.now())
        model.eval()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
        result = dict() 
        for i in range(len(datasets)):
            dataset = datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()
            # print("     Starting batch test - Soft voting", datetime.now())
            
            for batch_id, batch_data in enumerate(data_loader):
                avg_loss = 0
                avg_probabilities = list()
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # ls_prompt = [torch.sum(i) for i in model.client_local_prompts]
                # print("Sum prompts in testing", ls_prompt)
                # print("          Starting each ensembled model - Soft voting", datetime.now())
                for glo_pool in model.client_global_pools:
                    model.global_pool = glo_pool
                    # print(model.pool.prompt[model.pool.top_k_idx])
                    # print(torch.sum(local_prompt))
                    # import pdb; pdb.set_trace()
                    # print("               Starting 1 inference", datetime.now())
                    loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                    # print("               End 1 inference", datetime.now())
                    probabilities = F.softmax(outputs, dim=1)
                    # if avg_probabilities.device != probabilities.device:
                    #     avg_probabilities = avg_probabilities.to(probabilities.device)
                    avg_probabilities.append(probabilities)
                    avg_loss += loss
                    # print("          End each ensembled model - Soft voting", datetime.now())
                
                
                median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
                avg_loss /= len(model.client_global_pools)

                predicted_classes = torch.argmax(median_probabilities, dim=1)
                total_loss += avg_loss.item() * len(batch_data['label'])
                # import pdb; pdb.set_trace()
                predicts.extend(predicted_classes.cpu().tolist())
                # print("\tEnd batch test - Soft voting", datetime.now())
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End other server test - Soft voting", datetime.now())
            
        return result
    
    @torch.no_grad()
    def server_other_test_kmeans_median_soft_voting(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST_SOFT_VOTING-----------------------")
        # print("Starting other server test - Soft voting", datetime.now())
        model.eval()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
        result = dict() 
        for i in range(len(datasets)):
            dataset = datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()
            # print("     Starting batch test - Soft voting", datetime.now())
            
            # Get representative pools using clustering based on prompt similarity
            representative_pools = get_representative_pools(model, n_clusters=3)

            for batch_id, batch_data in enumerate(data_loader):
                avg_loss = 0
                avg_probabilities = list()
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # ls_prompt = [torch.sum(i) for i in model.client_local_prompts]
                # print("Sum prompts in testing", ls_prompt)
                # print("          Starting each ensembled model - Soft voting", datetime.now())
                for glo_pool in representative_pools:
                    model.global_pool = glo_pool
                    # print(model.pool.prompt[model.pool.top_k_idx])
                    # print(torch.sum(local_prompt))
                    # import pdb; pdb.set_trace()
                    # print("               Starting 1 inference", datetime.now())
                    loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                    # print("               End 1 inference", datetime.now())
                    probabilities = F.softmax(outputs, dim=1)
                    # if avg_probabilities.device != probabilities.device:
                    #     avg_probabilities = avg_probabilities.to(probabilities.device)
                    avg_probabilities.append(probabilities)
                    avg_loss += loss
                    # print("          End each ensembled model - Soft voting", datetime.now())
                
                
                median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
                avg_loss /= len(model.client_global_pools)

                predicted_classes = torch.argmax(median_probabilities, dim=1)
                total_loss += avg_loss.item() * len(batch_data['label'])
                # import pdb; pdb.set_trace()
                predicts.extend(predicted_classes.cpu().tolist())
                # print("\tEnd batch test - Soft voting", datetime.now())
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End other server test - Soft voting", datetime.now())
            
        return result


    @torch.no_grad()
    def server_other_test_agglo_median_soft_voting(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST_SOFT_VOTING-----------------------")
        # print("Starting other server test - Soft voting", datetime.now())
        model.eval()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
        result = dict() 
        for i in range(len(datasets)):
            dataset = datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()
            # print("     Starting batch test - Soft voting", datetime.now())
            
            # Get representative pools using clustering based on prompt similarity
            representative_pools = get_representative_pools_agglomerative(model, n_clusters=3)

            for batch_id, batch_data in enumerate(data_loader):
                avg_loss = 0
                avg_probabilities = list()
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # ls_prompt = [torch.sum(i) for i in model.client_local_prompts]
                # print("Sum prompts in testing", ls_prompt)
                # print("          Starting each ensembled model - Soft voting", datetime.now())
                for glo_pool in representative_pools:
                    model.global_pool = glo_pool
                    # print(model.pool.prompt[model.pool.top_k_idx])
                    # print(torch.sum(local_prompt))
                    # import pdb; pdb.set_trace()
                    # print("               Starting 1 inference", datetime.now())
                    loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                    # print("               End 1 inference", datetime.now())
                    probabilities = F.softmax(outputs, dim=1)
                    # if avg_probabilities.device != probabilities.device:
                    #     avg_probabilities = avg_probabilities.to(probabilities.device)
                    avg_probabilities.append(probabilities)
                    avg_loss += loss
                    # print("          End each ensembled model - Soft voting", datetime.now())
                
                
                median_probabilities, _ = torch.median(torch.stack(avg_probabilities), dim=0)
                avg_loss /= len(model.client_global_pools)

                predicted_classes = torch.argmax(median_probabilities, dim=1)
                total_loss += avg_loss.item() * len(batch_data['label'])
                # import pdb; pdb.set_trace()
                predicts.extend(predicted_classes.cpu().tolist())
                # print("\tEnd batch test - Soft voting", datetime.now())
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End other server test - Soft voting", datetime.now())
            
        return result

    @torch.no_grad()
    def server_other_test_agglo_mean_soft_voting(self, model, transformer, text_embeddings, datasets, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        # print("-------------------------SERVER_OTHER_TEST_SOFT_VOTING-----------------------")
        # print("Starting other server test - Soft voting", datetime.now())
        model.eval()
        other_test_datasets, names = datasets
        result = dict() 
        for i in range(len(other_test_datasets)):
            dataset = other_test_datasets[i]
        result = dict() 
        for i in range(len(datasets)):
            dataset = datasets[i]
            if batch_size==-1:batch_size=len(dataset)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            # for test_combi_index in range(len(2)):
            total_loss = 0.0
            labels = list()
            predicts = list()
            # print("     Starting batch test - Soft voting", datetime.now())
            
            # Get representative pools using clustering based on prompt similarity
            representative_pools = get_representative_pools_agglomerative(model, n_clusters=3)

            for batch_id, batch_data in enumerate(data_loader):
                avg_loss = 0
                avg_probabilities = list()
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data['label'])
                # ls_prompt = [torch.sum(i) for i in model.client_local_prompts]
                # print("Sum prompts in testing", ls_prompt)
                # print("          Starting each ensembled model - Soft voting", datetime.now())
                for glo_pool in representative_pools:
                    model.global_pool = glo_pool
                    # print(model.pool.prompt[model.pool.top_k_idx])
                    # print(torch.sum(local_prompt))
                    # import pdb; pdb.set_trace()
                    # print("               Starting 1 inference", datetime.now())
                    loss_leads, loss, outputs = model(transformer, text_embeddings, batch_data)
                    # print("               End 1 inference", datetime.now())
                    probabilities = F.softmax(outputs, dim=1)
                    # if avg_probabilities.device != probabilities.device:
                    #     avg_probabilities = avg_probabilities.to(probabilities.device)
                    avg_probabilities.append(probabilities)
                    avg_loss += loss
                    # print("          End each ensembled model - Soft voting", datetime.now())
                
                
                avg_probabilities = torch.mean(torch.stack(avg_probabilities), dim=0)
                avg_loss /= len(model.client_global_pools)

                predicted_classes = torch.argmax(avg_probabilities, dim=1)
                total_loss += avg_loss.item() * len(batch_data['label'])
                # import pdb; pdb.set_trace()
                predicts.extend(predicted_classes.cpu().tolist())
                # print("\tEnd batch test - Soft voting", datetime.now())
                # TO_DELETE
                # if batch_id==0:
                #     break
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result[names[i]+'_loss'] = total_loss / len(dataset)
            result[names[i]+'_acc'] = accuracy
        
        # print("End other server test - Soft voting", datetime.now())
            
        return result

@torch.no_grad()
def get_representative_pools(model, n_clusters=3):
    """
    Find representative pools from all client_global_pools by clustering them based on prompt similarity and selecting the closest to centroid.
    
    :param model: The model containing client_global_pools with prompt as nn.Parameter (2D).
    :param n_clusters: Number of clusters to form.
    :return: List of representative client_global_pools.
    """
    # Extract prompts from each global pool
    prompts = []
    for glo_pool in model.client_global_pools[1:]:
        prompts.append(glo_pool.prompt.detach().cpu().numpy())  # Convert prompts (nn.Parameter) to numpy array
    
    # Assume each prompt is a 2D array, we will flatten it for similarity calculation
    flattened_prompts = [prompt.flatten() for prompt in prompts]  # Flatten 2D prompt to 1D
    
    flattened_prompts = np.array(flattened_prompts)
    # Compute cosine similarity between all prompts
    similarity_matrix = cosine_similarity(flattened_prompts)  # Shape: (num_pools, num_pools)
    
    # Apply KMeans clustering based on similarity matrix
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    
    # Group pools by cluster and select the one closest to the centroid for each cluster
    pool_features = np.array(flattened_prompts)
    representatives = []
    representatives.append(model.client_global_pools[0])
    
    for i in range(n_clusters):
        # Find all pools in the current cluster
        group_indices = np.where(cluster_labels == i)[0]
        if len(group_indices) == 0:
            continue
        group_pools = pool_features[group_indices]
        # print([i.shape for i in group_pools])
        # Calculate the centroid of the cluster
        centroid = np.mean(group_pools, axis=0)
        # import pdb; pdb.set_trace()
        # Find the pool closest to the centroid
        closest_index = np.argmin(np.linalg.norm(group_pools - centroid, axis=1))
        
        # Append the representative (the closest pool to the centroid)
        representatives.append(model.client_global_pools[group_indices[closest_index] + 1])

    return representatives


@torch.no_grad()
def get_representative_pools_agglomerative(model, n_clusters=3):
    """
    Find representative pools from all client_global_pools by using Agglomerative Clustering based on prompt similarity.
    :param model: The model containing client_global_pools with prompt as nn.Parameter (2D).
    :param n_clusters: Number of clusters to form.
    :return: List of representative client_global_pools.
    """
    # Extract prompts from each global pool
    prompts = []
    for glo_pool in model.client_global_pools[1:]:
        prompts.append(glo_pool.prompt.detach().cpu().numpy())  # Convert prompts (nn.Parameter) to numpy array
    
    # Flatten the prompts (2D -> 1D)
    flattened_prompts = [prompt.flatten() for prompt in prompts]
    
    flattened_prompts = np.array(flattened_prompts)
    # Apply Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = agglomerative.fit_predict(flattened_prompts)
    
    # Group pools by cluster and select the one closest to the centroid for each cluster
    pool_features = np.array(flattened_prompts)
    representatives = []
    representatives.append(model.client_global_pools[0])
    
    for i in range(n_clusters):
        group_indices = np.where(cluster_labels == i)[0]
        group_pools = pool_features[group_indices]
        
        # Calculate the centroid of the cluster
        centroid = np.mean(group_pools, axis=0)
        
        # Find the pool closest to the centroid
        closest_index = np.argmin(np.linalg.norm(group_pools - centroid, axis=1))
        
        # Append the representative (the closest pool to the centroid)
        representatives.append(model.client_global_pools[group_indices[closest_index] + 1])

    return representatives


def plot_confusion_matrix(y_true, y_pred, model='server', current_round=-1, output_file=None, classes=None, normalize=True):
    """
    Plots the confusion matrix.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        output_file (str, optional): File to save the plot. If None, plot is not saved.
        classes (list, optional): List of class labels.
        normalize (bool, optional): Whether to normalize confusion matrix or not.
        
    Returns:
        None
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # If classes are not provided, use unique labels in y_true and y_pred
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Normalize confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    import pdb; pdb.set_trace()
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot if output_file is provided
    if output_file:
        plt.savefig(output_file)
    
    # Show plot
    # Convert the plot to an image
    plt_img = wandb.Image(plt)
    # Log the image to wandb
    wandb.log({"{}_confusion_matrix".format(model): plt_img}, step=current_round)
    plt.show()
    # Close plot to avoid memory leakage
    plt.close()