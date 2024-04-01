from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
from collections import Counter, OrderedDict, defaultdict
import dataloader
import time
# from koala_reward import PromptDebiasReward

class StereoSetDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        SAU: List[str],
        scs_id: List[str],
        BLANK: List[str],
        # S_id: List[str], 
        # A_id: List[str], 
        # U_id: List[str], 
        # S_text: List[str], 
        # A_text: List[str], 
        # U_text: List[str],
    ):
        assert len(source_texts) == len(SAU)
        self.source_texts = source_texts
        self.SAU = SAU
        self.scs_id = scs_id
        self.BLANK = BLANK
        # self.S_id = S_id
        # self.A_id = A_id
        # self.U_id = U_id
        # self.S_text = S_text
        # self.A_text = A_text
        # self.U_text = U_text


        

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        # print('idx------',idx)
        # print(self.S_text[idx])
        # time.sleep(200)
        # item = {
        #         'S_id': self.S_id[idx],
        #         'A_id': self.A_id[idx],
        #         'U_id': self.U_id[idx],
        #         'S_text': self.S_text[idx],
        #         'A_text': self.A_text[idx],
        #         'U_text': self.U_text[idx],
        #         }

        item = {'source_texts': self.source_texts[idx],
                'SAU': self.SAU[idx],
                'scs_id': self.scs_id[idx],
                'BLANK': self.BLANK[idx],}
        return item
    
def GetID_Context(clusters):
    # S_text, A_text, U_text =  [], [], []
    # S_id, A_id, U_id = [], [], []
    
    source_texts, SAU, scs_id, BLANK = [], [], [], []
    # input_file = os.path.join("/home/zhichao/qufeiyu/vi", "test.json")
    # input_file = os.path.join("/home/zhichao/qufeiyu/vi", "dev.json")
    # stereoset = dataloader.StereoSet(input_file)
    # clusters = stereoset.get_intrasentence_examples()
    id2gold = {}
    for example in clusters:
        for sentence in example.sentences:
            id2gold[sentence.ID] = sentence.gold_label
    for cluster in clusters:
        BLANK.append(cluster.context)
        BLANK.append(cluster.context)
        BLANK.append(cluster.context)
        for sentence in cluster.sentences:
            source_texts.append(sentence.sentence)
            scs_id.append(sentence.ID)
            if sentence.gold_label == 'stereotype':
                SAU.append('s')
            if sentence.gold_label == 'anti-stereotype':
                SAU.append('a')
            if sentence.gold_label == 'unrelated':
                SAU.append('u')
    # print('###########sleep in koala_helper.py#####')
    # time.sleep(20000)

    return source_texts, SAU, scs_id, BLANK
            


def make_stereoset_id_dataset(input_file) -> Tuple[StereoSetDataset]: 
    data_dict = {}
    
    for split in ['train', 'dev', 'test']: 
        source_texts, SAU, scs_id, BLANK = load_idlist_dataset(split, input_file)#3 lists
        
        data_dict[split] = StereoSetDataset(source_texts, SAU, scs_id, BLANK)

    return (data_dict['train'], data_dict['dev'], data_dict['test'])


def load_idlist_dataset(
    split: str,
    input_file: str,
    
    ) -> Tuple[List[str]]:
    print('split-----------------',split)
    assert split in ['train', 'dev', 'test']

    stereoset = dataloader.StereoSet(input_file)
    clusters = stereoset.get_intrasentence_examples()
    domain2example = {
        "intrasentence": defaultdict(lambda: []),
    }

    for example in clusters:
        domain2example["intrasentence"][example.bias_type].append(example)       
        
    clusters_profession = domain2example["intrasentence"]['profession']#2398
    clusters_gender = domain2example["intrasentence"]['gender']#771
    clusters_race = domain2example["intrasentence"]['race']#2976
    clusters_religion = domain2example["intrasentence"]['religion']#247
    clusters_all = stereoset.get_intrasentence_examples()

    clusters = clusters_all # choose which cluster
    
    
    # if split=='train':
    #     clusters = clusters[0:20]
    # if split=='dev':
    #     clusters = clusters[20:30]
    # if split=='test':
    #     clusters = clusters[30:]

    #Middle batch
    # if split=='train':
    #     clusters = clusters[0:128]
    # if split=='dev':
    #     clusters = clusters[128:128+32]
    # if split=='test':
    #     clusters = clusters[256:]

    # For testing why evalu is same tokens:
    test_step = 100
    General_number = 20
    if split=='train':
        clusters = clusters # train
        # clusters = clusters[0:1] # eval--3
        # clusters = clusters[0:General_number] # general
    if split=='dev':
        clusters = clusters[::test_step] # train
        # clusters = clusters # eval--19176
        # clusters = clusters[General_number :General_number*2] # general
    if split=='test':
        # clusters = clusters[256:]
        clusters = clusters[0:1]

    # if split=='train':
    #     clusters = clusters[0:num_c]
    # if split=='dev':
    #     clusters = clusters[num_c :num_c*2]
    # if split=='test':
    #     # clusters = clusters[256:]
    #     clusters = clusters[num_c*2:]

    source_texts, SAU, scs_id, BLANK = GetID_Context(clusters)
    print('Dataset source_texts length',len(source_texts))
    print('Dataset BLANK length',len(BLANK))
    # print('###########################sleep in koala_helper.py###########################')
    # time.sleep(20000000)
    #3:1
    return source_texts, SAU, scs_id, BLANK
    

