import functools
from .base_dataset import BaseDataset
# from base_dataset import BaseDataset        # run __main__
from torch.utils.data import DataLoader
import torch
import random, os
from datetime import datetime
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from collections import Counter

class IMDBDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        # assert split in ["train", "test"]
        self.split = split

        if split == "train":
            names = ["mmimdb_train"]
        else:
            names = ["mmimdb_test"]  
        # import pdb; pdb.set_trace()
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="plots",
            split=self.split,
            remove_duplicate=False,
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        # import pdb; pdb.set_trace()
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.','')
        missing_type = missing_info['type'][split]       
        both_ratio = missing_info['both_ratio']         # 0.5
        bratio = str(both_ratio).replace('.','')
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}_{bratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.table['image'])
        # import pdb; pdb.set_trace()
        
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1
                    missing_index_image  = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
            torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table


    def __getitem__(self, index):
        # index -> pair data index
        # image_index -> image index in table
        # question_index -> plot index in texts of the given image
        image_index, question_index = self.index_mapper[index]
        
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        simulate_missing_type = 0
        if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        
        # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            # import pdb; pdb.set_trace()
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        #missing text, dummy text is ''
        if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
            text = ''
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )   
            text = (text, encoding)
        else:
            text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        # import pdb; pdb.set_trace()
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        }

if __name__=='__main__':
    print(datetime.now(), "Start creating Datasets")
    data_dir = "../../benchmark/RAW_DATA/IMDB/generate_arrows"
    transform_keys = ['pixelbert']
    split="test"
    image_size = 384
    max_text_len = 1024
    draw_false_image = 0
    draw_false_text = 0
    image_only = False
    _config = {
        'missing_ratio':
            {'test': 1,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables/',
        'missing_type':
            {'test': 'text',
            'train': 'both'},
        'both_ratio': 0,
        'simulate_missing': False
    }
    missing_info = {
            'ratio' : _config["missing_ratio"],
            'type' : _config["missing_type"],
            'both_ratio' : _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'simulate_missing' : _config["simulate_missing"]
        }
        # for bash execution
    # if _config["test_ratio"] is not None:
    #     missing_info['ratio']['val'] = _config["test_ratio"]
    #     missing_info['ratio']['test'] = _config["test_ratio"]
    # if _config["test_type"] is not None:
    #     missing_info['type']['val'] = _config["test_type"]
    #     missing_info['type']['test'] = _config["test_type"]
            
    train_dataset = IMDBDataset(data_dir, transform_keys, split=split, 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
    train_dataset.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')

    collator = DataCollatorForLanguageModeling

    train_dataset.mlm_collator = collator(tokenizer=train_dataset.tokenizer, mlm=True, mlm_probability=0.15)
    
    train_dataset.collate = functools.partial(train_dataset.collate, mlm_collator=train_dataset.mlm_collator)

    # import pdb; pdb.set_trace()
    missing_types = []
    labels = []
    for i in range(len(train_dataset)):
        data_sample = train_dataset[i]
        missing_type = data_sample["missing_type"]
        missing_types.append(missing_type)
        label = data_sample["label"]
        labels.append(label)

    dict_types = Counter(missing_types)
    dict_labels = Counter(labels)
    str_ = '\t' + str({k: dict_types[k] for k in sorted(dict_types)}) + '\t\t' + str({k: dict_labels[k] for k in sorted(dict_labels)})
    print(str_)
    exit()
    # import pdb; pdb.set_trace()

    batch_size = 32
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=train_dataset.collate)
    batch = next(iter(train_dataloader))
    
    # img, text, label, missing_type = batch['image'], batch['text'], batch['label'], batch['missing_type']
    import pdb; pdb.set_trace()

    # print(label.shape, label)
    # print(missing_type)
    # for k,v in img.items():
    #     print("Image sample", k, v.shape)
    # for k,v in text.items():
    #     print("Image sample", k, v.shape)

    # import pdb; pdb.set_trace()