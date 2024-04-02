#################################################################################
import os
import json
import pickle
import string
from tqdm import tqdm
from pprint import pprint
from string import punctuation
from typing import List, Optional, Tuple
# from bias_bench.benchmark.stereoset import dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import time

#################################################################################
import itertools

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import Counter, OrderedDict, defaultdict
from rlprompt.rewards import BaseReward

import logging
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='prompt.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

#-------------------- CLOZE MBPS------------------------------------------------------
def GetIndex(tokenizer, source_texts_3):
    source_texts_3_token = [tokenizer.encode(x) for x in source_texts_3] 
    front_index = 0
    while source_texts_3_token[0][front_index] == source_texts_3_token[1][front_index] == source_texts_3_token[2][front_index]:
        front_index += 1
    source_texts_3_token_reverse = [x[::-1] for x in source_texts_3_token] 
    rear_index = 0
    while source_texts_3_token_reverse[0][rear_index] == source_texts_3_token_reverse[1][rear_index] == source_texts_3_token_reverse[2][rear_index]:
        rear_index += 1
    index = []
    for idx in range(3):
        Index = list(range(len(source_texts_3_token[idx])))
        F = front_index
        R = len(source_texts_3_token[idx])-rear_index
        index.append(Index[F-1:R-1])
    return index


def GetProb(model, tokenizer,  source_texts_3, scs_id_3, BLANK, GeneratedPrompt, device):
    BLANK = BLANK[0]
    Index = GetIndex(tokenizer, source_texts_3)
    ManualPrompt = 'Here is full context with BLANK: '
    Appending = ManualPrompt + BLANK + GeneratedPrompt
    print('Appending: ', Appending)
    unconditional_start_token = ''
    start_token = (
        torch.tensor(tokenizer.encode(unconditional_start_token))
        .to(device)
        .unsqueeze(0)
    )
    with torch.no_grad():
        initial_token_probabilities = model(start_token)
    initial_token_probabilities = torch.softmax(
        initial_token_probabilities[0], dim=-1
    )
    assert initial_token_probabilities.shape[0] == 1
    assert initial_token_probabilities.shape[1] == 1
    predictions = []
    joint_sentence_probability = []
    for i, sentence in enumerate(source_texts_3):
#         print('Sentence:', sentence)
        probabilities = {}
        print('Input: ', Appending + sentence)
        tokens = tokenizer.encode(sentence)
        Appending_tokens = tokenizer.encode(Appending)
#         print('Appending_tokens',Appending_tokens)
#         print('sentence tokens',tokens)
        mixed_tokes = (Appending_tokens + tokens)
        del mixed_tokes[len(Appending_tokens)]
#         print('mixed_tokes',mixed_tokes)
        tokens_tensor = torch.tensor(mixed_tokes).to(device).unsqueeze(0)
#         print('tokens_tensor.shape',tokens_tensor.shape)
        with torch.no_grad():
            joint_sentence_probability = [
                initial_token_probabilities[0, 0, mixed_tokes[0]].item()# tokens[0] is <s>, tokens[1] is the first token in sentence
            ]
            output = torch.softmax(model(tokens_tensor)[0], dim=-1)
        #                 print('output',output)
#             print('output.shape',output.shape)#output.shape torch.Size([1, 50, 32000])


        for idx in range(1, len(mixed_tokes)): # different len(tokens) (llama is 1 more than gpt2)
            joint_sentence_probability.append(
                output[0, idx - 1, mixed_tokes[idx]].item()#这个idx-1可能是因为model的output的idx-1是idx的概率
            )
        assert len(mixed_tokes) == len(joint_sentence_probability)
        m = len(Appending_tokens)  # 列表长度
        # print('original',joint_sentence_probability)
        joint_sentence_probability =  joint_sentence_probability[m:]
        # print('croped joint_sentence_probability',joint_sentence_probability)
        # print('croped joint_sentence_tokens',tokenizer.decode(mixed_tokes[m:]))

        
        assert len(tokens)-1 == len(joint_sentence_probability)
#         print('joint_sentence_probability',joint_sentence_probability)
#         print('Crop mixed_tokes',mixed_tokes[m:])
        index = Index[i]
        subword_tokes = [mixed_tokes[m:][i] for i in index]
        
        joint_subword_probability = [joint_sentence_probability[i] for i in index]
        # print('subword_tokes',tokenizer.decode(subword_tokes))
        # print('joint_subword_probability',joint_subword_probability)
        
#-------------------- MBPS token ------------------------------------------------------
        # print('MBPS token')
        # score = np.sum([np.log2(i) for i in joint_subword_probability])
        # score /= len(joint_subword_probability)

#-------------------- MBPS sentence ------------------------------------------------------
        print('MBPS sentence')
        score = np.sum([np.log2(i) for i in joint_sentence_probability])
        score /= len(joint_sentence_probability)
#--------------------------------------------------------------------------
        
        score = np.power(2, score)
        probabilities["id"] = scs_id_3[i]
        probabilities["score"] = score
        predictions.append(probabilities)
    return predictions


def GetScore_orderSAU(model, tokenizer, source_texts_3, SAU_3, scs_id_3, BLANK, instruction, device):
    rst = []
    x  = GetProb(model, tokenizer,  source_texts_3, scs_id_3, BLANK, instruction, device)
    for i in range (3):
        if  SAU_3[i] == 's':
            x_stereotype_score = x[i]['score']
        if  SAU_3[i] == 'a':
            x_antistereotype_score = x[i]['score']
        if  SAU_3[i] == 'u':
            x_unrelated_score = x[i]['score']
    
    rst.append(x_stereotype_score)
    rst.append(x_antistereotype_score)
    rst.append(x_unrelated_score)
    # With order : s,a,u
    return rst, x

#-------------------- CLOZE ------------------------------------------------------

def Inference3OutputBiasScore(model, tokenizer, source_texts_3, scs_id_3, instruction,device):
    tem = 'before'
    # tem = 'after' 
    # unconditional_start_token = '' #Vicuna stasrt token
    # start_token torch.Size([1, 0]) tensor([], device='cuda:0', size=(1, 0))
    unconditional_start_token = '<|endoftext|>' #GPT2 stasrt token
    # start_token torch.Size([1, 1]) tensor([[50256]], device='cuda:0')

    start_token = (
        torch.tensor(tokenizer.encode(unconditional_start_token))
        .to(device)
        .unsqueeze(0)
    )

    print('start_token',start_token.shape,start_token)

    with torch.no_grad():
        initial_token_probabilities = model(start_token)
    initial_token_probabilities = torch.softmax(
        initial_token_probabilities[0], dim=-1
    )
    assert initial_token_probabilities.shape[0] == 1
    assert initial_token_probabilities.shape[1] == 1

    if tem == 'after':
        print('after')
        predictions = []
        joint_sentence_probability = []
        for i, sentence in enumerate(source_texts_3):
            probabilities = {}
            # print('Input: ',sentence + instruction)
            tokens = tokenizer.encode(sentence)
            instruction_tokens = tokenizer.encode(instruction)
            # print('tokens',tokens)
            # print('instruction_tokens',instruction_tokens)
            mixed_tokes = (tokens + instruction_tokens)
            del mixed_tokes[len(tokens)]
            # print('mixed_tokes',mixed_tokes)
            tokens_tensor = torch.tensor(mixed_tokes).to(device).unsqueeze(0)
            with torch.no_grad():
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, mixed_tokes[0]].item()# tokens[0] is <s>, tokens[1] is the first token in sentence
                ]
                output = torch.softmax(model(tokens_tensor)[0], dim=-1)

            for idx in range(1, len(mixed_tokes)): # different len(tokens) (llama is 1 more than gpt2)
                joint_sentence_probability.append(
                    output[0, idx - 1, mixed_tokes[idx]].item()#这个idx-1可能是因为model的output的idx-1是idx的概率
                )
            assert len(mixed_tokes) == len(joint_sentence_probability)
            n = len(tokens)  # 列表长度
            # print('original',joint_sentence_probability)
            joint_sentence_probability =  joint_sentence_probability[1:n]
            # print('croped  ',joint_sentence_probability)

            assert len(tokens)-1 == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability])
            score /= len(joint_sentence_probability)
            score = np.power(2, score)
            probabilities["id"] = scs_id_3[i]
            probabilities["score"] = score
            predictions.append(probabilities)

    if tem == 'before':
        print('before')
        predictions = []
        joint_sentence_probability = []
        for i, sentence in enumerate(source_texts_3):
            probabilities = {}
            # print('Input: ', instruction + sentence)
            tokens = tokenizer.encode(sentence)
            instruction_tokens = tokenizer.encode(instruction)
            # print('tokens',tokens)
            # print('instruction_tokens',instruction_tokens)
            mixed_tokes = (instruction_tokens + tokens)
            del mixed_tokes[len(instruction_tokens)]
            # print('mixed_tokes',mixed_tokes)
            tokens_tensor = torch.tensor(mixed_tokes).to(device).unsqueeze(0)
            with torch.no_grad():
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, mixed_tokes[0]].item()# tokens[0] is <s>, tokens[1] is the first token in sentence
                ]
                output = torch.softmax(model(tokens_tensor)[0], dim=-1)

            for idx in range(1, len(mixed_tokes)): # different len(tokens) (llama is 1 more than gpt2)
                joint_sentence_probability.append(
                    output[0, idx - 1, mixed_tokes[idx]].item()#这个idx-1可能是因为model的output的idx-1是idx的概率
                )
            assert len(mixed_tokes) == len(joint_sentence_probability)
            m = len(instruction_tokens)  # 列表长度
            # print('original',joint_sentence_probability)
            joint_sentence_probability =  joint_sentence_probability[m:]
            # print('croped  ',joint_sentence_probability)


            assert len(tokens)-1 == len(joint_sentence_probability)
            score = np.sum([np.log2(i) for i in joint_sentence_probability])
            score /= len(joint_sentence_probability)
            score = np.power(2, score)
            probabilities["id"] = scs_id_3[i]
            probabilities["score"] = score
            predictions.append(probabilities)
    return predictions


def GetScore(model, tokenizer, source_texts_3, SAU_3, scs_id_3, instruction, device):
    rst = []
    x  = Inference3OutputBiasScore(model, tokenizer,  source_texts_3, scs_id_3, instruction, device)
    for i in range (3):
        if  SAU_3[i] == 's':
            x_stereotype_score = x[i]['score']
        if  SAU_3[i] == 'a':
            x_antistereotype_score = x[i]['score']
        if  SAU_3[i] == 'u':
            x_unrelated_score = x[i]['score']
    
    rst.append(x_stereotype_score)
    rst.append(x_antistereotype_score)
    rst.append(x_unrelated_score)
    # With order : s,a,u
    return rst, x


class PromptDebiasReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        
        # is_mask_lm: Optional[bool],#False
        # compute_zscore: bool,
        # incorrect_coeff: float, # lambda_1 in paper
        # correct_coeff: float, # lambda_2 in paper
        # num_classes: int,
        # verbalizers: List[str],
        # template: Optional[str]
    ):
        super().__init__()
        self.num_repeats =  4 # Num of repetitions for each example
        self.num_bootstraps = 4

        self.ALPHA = 0
        # self.ALPHA = 0.5
        # self.ALPHA = 0.2
        # self.ALPHA = 0.4
        # self.ALPHA = 0.6
        # self.ALPHA = 0.8
        # self.ALPHA = 1





        # self.task_lm = "samwit/koala-7b"
        # self.task_lm = "eachadea/vicuna-7b-1.1"
        self.task_lm = "openai-community/gpt2"



        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        print('main DEVICE::::::::', self.device)
        ##########################################                           
        print('begin to load tokenizer...')
        t1 = time.time()                          
        # self._tokenizer = AutoTokenizer.from_pretrained(
        #     self.task_lm, cache_dir='/raid/zhichao/qufeiyu', local_files_only=True)
        # self._tokenizer = LlamaTokenizer.from_pretrained(
        #     self.task_lm,  cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model', local_files_only=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm,  
                                                        cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model', 
                                                        local_files_only=True)
        t2 = time.time()      
        print('tokenizer load success. Time:',(t2-t1))
        print('begin to load generator...')
        
        # self._generator = (LlamaForCausalLM.from_pretrained(
        #                     self.task_lm,  cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model', local_files_only=True)
        #                    .to(self.device))  
        self._generator = (GPT2LMHeadModel.from_pretrained(
                            self.task_lm,  cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model', local_files_only=True)
                           .to(self.device))

        # self._generator = (AutoModelForCausalLM.from_pretrained(
        #                     self.task_lm, cache_dir='/raid/zhichao/qufeiyu', local_files_only=True)
        #                    )

        print('main DEVICE::::::::', self.device)
        t3 = time.time()  
        print('generator load success. Time:',(t3-t2)) 
        ##########################################                           

        
        # self.incorrect_coeff = incorrect_coeff
        # self.correct_coeff = correct_coeff
        self._counter = 0

    def load_default_template(self) -> str:
        template = "{prompt} {sentence_1}"
        return template

    def forward(
        self,
        source_texts: List[str],
        SAU: List[str],
        scs_id: List[str],
        BLANK: List[str],
        
        # source_texts: List[str],
        # class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        assert mode in ["train", "infer"]
        
        if mode == "train":
            self.num_repeats =  20
            self._counter += 1
        if mode == "infer":
            self.num_repeats =  1
            # eval_scores = []
        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        print('prompt_strings',prompt_strings)
        # ['ĠImageĠByImageHeightMedium', 'SetStudyFireActionDistance', 'DemocraticTextDetailsBatteryDetails', 'ĠFormerTextPolicyDescriptionFile']
        logging.info(f'prompt_strings:::{prompt_strings}')
        batch_size = len(source_texts)#3
        # print('batch_size = len(source_texts)',batch_size)
        # print('BEFORE REPEAT source_texts---------------',source_texts)
        # print('BEFORE REPEAT SAU---------------',SAU)
        # print('BEFORE REPEAT scs_id---------------',scs_id)
        # print('####################SLEEP on reward#########################')
        # time.sleep(15000)
        # source_texts = self._repeat_texts(source_texts)# [s,a,u,  s,a,u,  s,a,u,  s,a,u]
        # SAU = self._repeat_texts(SAU)# [s,a,u,  s,a,u,  s,a,u,  s,a,u]
        # scs_id = self._repeat_texts(scs_id)# [s,a,u,  s,a,u,  s,a,u,  s,a,u]
        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        # print('AFTER REPEAT source_texts---------------',source_texts)
        # print('AFTER REPEAT SAU---------------',SAU)
        # print('AFTER REPEAT scs_id---------------',scs_id)
        # print('eval test: BLANK---------------',BLANK)
        # print('####################SLEEP on reward#########################')
        # time.sleep(15000)
        #######################################################################
        # ta=time.time()
        # source_strs = self._repeat_texts(source_texts)
        # print('eval test: ', len(prompt_strings))#4
        # print('eval test: ', batch_size) #3


        #----------------- Condi ---------------------------------------------------
        for i, instruction in enumerate(prompt_strings): #general-8 condi-20
            reward: List[torch.Tensor] = []
            s_score, a_score, u_score = [], [], []
            print('\nTimes: ', self._counter, '| Prompt_No.',i , '|', instruction )
            gap_rewards = []
            print('source_texts_3--------',source_texts, 'SAU_3--------', SAU)
            rst, x = GetScore(self._generator, self._tokenizer, source_texts, SAU, scs_id, instruction, self.device) # Sentence socre
            # rst, x =  GetScore_orderSAU(self._generator, self._tokenizer, source_texts, SAU, scs_id, BLANK, instruction, self.device) # Subtoken socre
            #append instruction before context. if after threr will be no influence.
            print('Result--Ps:Pa:Pu------:',rst)

            #-----------------eval block 1/2--------------------------------------------------------------------------------
            # result_path = "/share/home/wenqingchen/feiyu/RL_debias/result"
            # #$1
            # # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis0.pkl'
            # # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis02.pkl'
            # # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis04.pkl'
            # # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis06.pkl'
            # # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis08.pkl'
            # result_name = '_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.pkl'

            # print('result_name:', result_name)

            # file_path = os.path.join(result_path, result_name)
            # if os.path.exists(file_path):
            #     with open(file_path, "rb") as file:
            #         data = pickle.load(file)
            #         data.extend(x)
            # else:
            #     data = x
            # with open(file_path, "wb") as file:
            #     pickle.dump(data, file)
                
            #-----------------eval block 1/2--------------------------------------------------------------------------------

            Ps = rst[0]
            s_score.append(Ps)
            Pa = rst[1]
            a_score.append(Pa)
            Pu = rst[2]
            u_score.append(Pu)

            ss = Ps/(Pa+Ps)
            lms = (Ps+Pa)/(Ps+Pa+2*Pu)
            icat = lms*(min(ss, 1-ss)/0.5)

            # Reward = icat
            Reward = self.ALPHA * lms + (1-self.ALPHA) * (min(ss, 1-ss)/0.5)

            print('ss--------',ss,'lms--------',lms,'icat--------',icat,'Alpha--------', self.ALPHA,'Reward--------',Reward)

            gap_rewards.append(Reward)
            gap_rewards = torch.Tensor(gap_rewards).to(self.device)
            reward = gap_rewards.mean().detach() * 100   #one prompt-one cluster
            
            quantities_to_log['gap_reward'].append(reward.item())
            rewards.append(reward)#one prompt-all repeated clusters

            # keep track of rewards for z-score normalization
            input_rewards['z'] += [reward.item()]
            print_strs = ['StereosetScore:-----', ss ]
            print_strs += ['LMScore:-----', lms ]
            print_strs += ['Reward-ICAT:-----', round(reward.item(), 2)]
            print(*print_strs)
        #----------------- Condi ---------------------------------------------------

        #----------------- General ---------------------------------------------------
        # for i, instruction in enumerate(prompt_strings): #general-8 condi-20
        #     reward: List[torch.Tensor] = []
        #     s_score, a_score, u_score = [], [], []
        #     print('\nTimes: ', self._counter, '| Prompt_No.',i , '|', instruction )
        #     gap_rewards = []
        #     for j in range(0, batch_size, 3):#  
        #         source_texts_3 = source_texts[j:j+3]
        #         SAU_3 = SAU[j:j+3]
        #         scs_id_3 = scs_id[j:j+3]
        #         BLANK_3 = BLANK[j:j+3]
        #         print('\n j, j+1, j+2--------',j, j+1, j+2)
        #         print('source_texts_3--------',source_texts_3, 'SAU_3--------', SAU_3)
        #         # rst, x = GetScore(self._generator, self._tokenizer, source_texts, SAU, scs_id, instruction, self.device) # Sentence socre
        #         #MBPS
        #         rst, x =  GetScore_orderSAU(self._generator, self._tokenizer, source_texts_3, SAU_3, scs_id_3, BLANK_3, instruction, self.device) # Subtoken socre
        #         #append instruction before context. if after threr will be no influence.
        #         print('Result--Ps:Pa:Pu------:',rst)
        #         Ps = rst[0]
        #         s_score.append(Ps)
        #         Pa = rst[1]
        #         a_score.append(Pa)
        #         Pu = rst[2]
        #         u_score.append(Pu)
        #         ss = Ps/(Pa+Ps)
        #         lms = (Ps+Pa)/(Ps+Pa+2*Pu)
        #         icat = lms*(min(ss, 1-ss)/0.5)
        #         print('ss--------',ss,'lms--------',lms,'icat--------',icat)
        #         gap_rewards.append(icat)
        #     gap_rewards = torch.Tensor(gap_rewards).to(self.device)
        #     print(i, 'th prompt: gap_rewards--------',gap_rewards)
        #     reward = gap_rewards.mean().detach() * 100 
        #     print('1prompt reward',reward)
        #     quantities_to_log['gap_reward'].append(reward.item())
        #     rewards.append(reward)#one prompt-all repeated clusters
        #     # keep track of rewards for z-score normalization
        #     input_rewards['z'] += [reward.item()]

        #     # Print examples
        #     # print_strs = ['Times: ', self._counter, '|', instruction, '\n']#for every instructions:
        #     # print_strs += ['Stereotype Score:', s_score]
        #     # print_strs += ['Anti-Stereotype Score:', a_score]
        #     # print_strs += ['Unrelated Score:', u_score]

        #     # print_strs = ['StereosetScore:-----', ss ]
        #     # print_strs += ['LMScore:-----', lms ]
        #     # print_strs += ['Reward-ICAT:-----', round(reward.item(), 2)]
        #     # print(*print_strs)
            
        #----------------- General ---------------------------------------------------
        print('8prompts rewards', rewards)


        rewards_tensor = torch.stack(rewards)
        print('rewards_tensor',rewards_tensor)


        # z-score normalization (2nd stage)
        if mode == 'train' :
            input_reward_means = {k: np.mean(v)
                                  for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v)
                                 for k, v in input_rewards.items()}
            # not source strings
            idx_means = torch.tensor(input_reward_means['z']).float()
            idx_stds = torch.tensor(input_reward_stds['z']).float()
            # print('rewards_tensor without z-score', rewards_tensor)
            print('rewards_tensor BEFORE', rewards_tensor)
            rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)
            print('rewards_tensor AFTER z-score', rewards_tensor)
            # time.sleep(20000)
            for i in range(rewards_tensor.size(0)):
                quantities_to_log['resized_reward'].append(
                    rewards_tensor[i].item())
            # print('Sleep here')
            # time.sleep(200000)
        elif mode == 'infer':  # Optional: Predict Val Prompts
            #############################################ERROR#############################################
            score = rewards_tensor.mean().item()
            # eval_scores.append(score)
            print('Our Instruction:')
            print(prompt_strings, score)
        ############################################# /10 #############################################    
        # rewards_tensor = rewards_tensor/10
        ############################################# /10 #############################################    
        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s)
                for s in tokens]
    def _repeat_texts(
        self,
        texts: List[str],
        num_repeats: Optional[int] = None
    ) -> List[str]:
        if num_repeats is None:
            num_repeats = self.num_repeats
        x = list(itertools.chain(*[[s for _ in range(num_repeats)]
                                      for s in texts]))
        return x[::num_repeats]*num_repeats


