# #################################################################################
# import os
# import json
# import pickle
# import string
# from tqdm import tqdm
# from pprint import pprint
# from string import punctuation
# from typing import List, Optional, Tuple
# # from bias_bench.benchmark.stereoset import dataloader
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cosine
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
# import time

# #################################################################################

# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
# from typing import List, Dict, Optional, Tuple, Union, Any
# from collections import Counter, OrderedDict, defaultdict
# from rlprompt.rewards import BaseReward

# import logging
# logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
#                     filename='prompt.log',
#                     filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     #a是追加模式，默认如果不写的话，就是追加模式
#                     format=
#                     '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     #日志格式
#                     )

# def Inference3OutputBiasScore(model, tokenizer, source_texts_3, scs_id_3, instruction,device):
#     unconditional_start_token = ''
#     start_token = (
#         torch.tensor(tokenizer.encode(unconditional_start_token))
#         .to(device)
#         .unsqueeze(0)
#     )
#     with torch.no_grad():
#         initial_token_probabilities = model(start_token)
#     initial_token_probabilities = torch.softmax(
#         initial_token_probabilities[0], dim=-1
#     )
#     assert initial_token_probabilities.shape[0] == 1
#     assert initial_token_probabilities.shape[1] == 1

#     predictions = []
#     joint_sentence_probability = []
#     for i, sentence in enumerate(source_texts_3):
#         probabilities = {}
#         print('Input: ',instruction+sentence)
#         tokens = tokenizer.encode(sentence)
#         instruction_tokens = tokenizer.encode(instruction)
#         mixed_tokes = (instruction_tokens + tokens)
#         del mixed_tokes[len(instruction_tokens)]
#         tokens_tensor = torch.tensor(mixed_tokes).to(device).unsqueeze(0)
#         with torch.no_grad():
#             joint_sentence_probability = [
#                 initial_token_probabilities[0, 0, mixed_tokes[0]].item()# tokens[0] is <s>, tokens[1] is the first token in sentence
#             ]
#             output = torch.softmax(model(tokens_tensor)[0], dim=-1)

#         for idx in range(1, len(mixed_tokes)): # different len(tokens) (llama is 1 more than gpt2)
#             joint_sentence_probability.append(
#                 output[0, idx - 1, mixed_tokes[idx]].item()#这个idx-1可能是因为model的output的idx-1是idx的概率
#             )
#         assert len(mixed_tokes) == len(joint_sentence_probability)
#         m = len(instruction_tokens)  # 列表长度
#         joint_sentence_probability =  joint_sentence_probability[m:]
#         assert len(tokens)-1 == len(joint_sentence_probability)
#         score = np.sum([np.log2(i) for i in joint_sentence_probability])
#         score /= len(joint_sentence_probability)
#         score = np.power(2, score)
#         probabilities["id"] = scs_id_3[i]
#         probabilities["score"] = score
#         predictions.append(probabilities)
#     return predictions
        
# def GetScore(model, tokenizer, source_texts_3, SAU_3, scs_id_3, instruction, device):
#     rst = []
#     x  = Inference3OutputBiasScore(model, tokenizer,  source_texts_3, scs_id_3, instruction, device)
#     for i in range (3):
#         if  SAU_3[i] == 's':
#             x_stereotype_score = x[i]['score']
#         if  SAU_3[i] == 'a':
#             x_antistereotype_score = x[i]['score']
#         if  SAU_3[i] == 'u':
#             x_unrelated_score = x[i]['score']
    
#     rst.append(x_stereotype_score)
#     rst.append(x_antistereotype_score)
#     rst.append(x_unrelated_score)
#     # With order : s,a,u
#     return rst

# class PromptDebiasReward(BaseReward):
#     def __init__(
#         self,
#         task_lm: str,
#         # is_mask_lm: Optional[bool],#False
#         # compute_zscore: bool,
#         # incorrect_coeff: float, # lambda_1 in paper
#         # correct_coeff: float, # lambda_2 in paper
#         # num_classes: int,
#         # verbalizers: List[str],
#         # template: Optional[str]
#     ):
#         super().__init__()
#         self.task_lm = "samwit/koala-7b"
#         self.device = torch.device("cuda" if torch.cuda.is_available()
#                                    else "cpu")
#         print('main DEVICE::::::::', self.device)
#         ##########################################                           
#         print('begin to load tokenizer...')
#         t1 = time.time()                          
#         self._tokenizer = AutoTokenizer.from_pretrained(
#             self.task_lm, cache_dir='/raid/zhichao/qufeiyu', local_files_only=True)
#         t2 = time.time()      
#         print('tokenizer load success. Time:',(t2-t1))
#         print('begin to load generator...')
        
#         # model = nn.DataParallel(model, device_ids=[0, 1])
#         self._generator = (AutoModelForCausalLM.from_pretrained(
#                             self.task_lm, cache_dir='/raid/zhichao/qufeiyu', local_files_only=True)
#                            .to(self.device))

#         # self._generator = (AutoModelForCausalLM.from_pretrained(
#         #                     self.task_lm, cache_dir='/raid/zhichao/qufeiyu', local_files_only=True)
#         #                    )
#         # self._generator = nn.DataParallel(self._generator, device_ids=[0,1])
#         # self._generator.to(self.device)
#         print('main DEVICE::::::::', self.device)
#         t3 = time.time()  
#         print('generator load success. Time:',(t3-t2)) 
#         ##########################################                           

        
#         # self.incorrect_coeff = incorrect_coeff
#         # self.correct_coeff = correct_coeff
#         self._counter = 0

#     def load_default_template(self) -> str:
#         template = "{prompt} {sentence_1}"
#         return template

#     def forward(
#         self,
#         source_texts: List[str],
#         SAU: List[str],
#         scs_id: List[str],
        
#         # source_texts: List[str],
#         # class_labels: List[int],
#         output_tokens: List[List[str]],
#         to_tensor: bool,
#         mode: str
#     ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
#         assert mode in ["train", "infer"]
        
#         if mode == "train":
#             self._counter += 1
#         # Process prompts and verbalizer indices
#         prompt_tokens = output_tokens
#         prompt_strings = self._convert_tokens_to_string(prompt_tokens)#16
#         logging.info(f'prompt_strings:::{prompt_strings}')
#         batch_size = len(source_texts)

#         rewards: List[torch.Tensor] = []
#         input_rewards: Dict[str, List[float]] = defaultdict(list)
#         quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
#         # print('prompt_strings---------------',prompt_strings)
#         # print('source_texts---------------',source_texts)
#         # print('batch_size---------------',batch_size)#30
#         #######################################################################
#         # ta=time.time()
#         for i, instruction in enumerate(prompt_strings): #16 every instruction like 'ResearchersCatalogAlertStudentSocial'
#             reward: List[torch.Tensor] = []
#             s_score, a_score, u_score = [], [], []
#             for j in range(0, batch_size, 3):#  20*3
#                 # tc = time.time()
#                 gap_rewards = []
#                 source_texts_3 = source_texts[j:j+3]
#                 SAU_3 = SAU[j:j+3]
#                 scs_id_3 = scs_id[j:j+3]
#                 print('j, j+1, j+2--------',j, j+1, j+2)
#                 print('source_texts_3--------',source_texts_3)
#                 print('SAU_3--------',SAU_3)

#                 rst = GetScore(self._generator, self._tokenizer, source_texts_3, SAU_3, scs_id_3, instruction, self.device)
                
#                 print('rst--------',rst)

#                 Ps = rst[0]
#                 s_score.append(Ps)
#                 Pa = rst[1]
#                 a_score.append(Pa)
#                 Pu = rst[2]
#                 u_score.append(Pu)

#                 ss = Ps/(Pa+Ps)
#                 lms = (Ps+Pa)/(Ps+Pa+2*Pu)
#                 icat = lms*(min(ss, 1-ss)/0.5)
#                 print('ss--------',ss)
#                 print('lms--------',lms)
#                 print('icat--------',icat)
#                 gap_rewards.append(icat)
#                 # td = time.time()
#                 # print('one cluster reward time:', td-tc)
#             gap_rewards = torch.Tensor(gap_rewards).to(self.device)
#             print('gap_rewards--------',gap_rewards)
#             reward = gap_rewards.mean().detach()    
            
#             quantities_to_log['gap_reward'].append(reward.item())
#             rewards.append(reward)

#             # keep track of rewards for z-score normalization
#             input_rewards['z'] += [reward.item()]

#             # Print examples
#             print_strs = ['Times: ', self._counter, '|', instruction, '\n']#for every instructions:

#             print_strs += ['Stereotype Score:', s_score]
            
#             print_strs += ['Anti-Stereotype Score:', a_score]
#             print_strs += ['Unrelated Score:', u_score]

#             print_strs += ['StereosetScore:-----', ss ]
#             print_strs += ['LMScore:-----', lms ]
#             print_strs += ['Reward-ICAT:-----', round(reward.item(), 2)]
#             print(*print_strs)
#         # tb=time.time()
#         # print('one batch reward time:', tb-ta)
#         rewards_tensor = torch.stack(rewards)

#         # z-score normalization (2nd stage)
#         if mode == 'train' :
#             input_reward_means = {k: np.mean(v)
#                                   for k, v in input_rewards.items()}
#             input_reward_stds = {k: np.std(v)
#                                  for k, v in input_rewards.items()}
#             # not source strings
#             idx_means = torch.tensor(input_reward_means['z']).float()
#             idx_stds = torch.tensor(input_reward_stds['z']).float()
#             print('rewards_tensor without z-score', rewards_tensor)
#             # print('rewards_tensor BEFORE', rewards_tensor)
#             # rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)
#             # print('rewards_tensor AFTER', rewards_tensor)
#             for i in range(rewards_tensor.size(0)):
#                 quantities_to_log['resized_reward'].append(
#                     rewards_tensor[i].item())
#             # print('Sleep here')
#             # time.sleep(200000)
#         elif mode == 'infer':  # Optional: Predict Val Prompts
#             score = rewards_tensor.mean().item()
#             print('Our Instruction:')
#             print(prompt_strings, score)

#         rewards_log = dict(
#             (reward_key, torch.mean(torch.tensor(reward_vals)))
#             for reward_key, reward_vals in quantities_to_log.items())

#         if to_tensor is True:
#             return rewards_tensor, rewards_log
#         else:
#             return rewards_tensor.tolist(), rewards_log

#     def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
#         return [self._tokenizer.convert_tokens_to_string(s)
#                 for s in tokens]


