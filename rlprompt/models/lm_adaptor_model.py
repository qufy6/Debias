import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer, AutoModel, GPT2LMHeadModel

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits

import time 
SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']

LM_HIDDEN_SIZES = {'distilgpt2': 768,
                   'gpt2': 768,
                   'gpt2-medium': 1024,
                   'gpt2-large': 1280,
                   'gpt2-xl': 1600}

from scipy.special import kl_div

import torch.nn.functional as F

def kl_regularization(logits, ori_logits, temperature=1.0):
    
    p = F.softmax(logits/temperature, dim=-1)
    q = F.softmax(ori_logits/temperature, dim=-1)
    kl_div = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=-1)

    return kl_div



class LMAdaptorModel(BaseModel):
    """Uses an MLP to modify the hidden states of an pre-trained LM

    The modified hidden state can then be passed into the original LM head
    to obtain output token logits. 
    
    Inspired by Houlsby et al. (2019): https://arxiv.org/abs/1902.00751
    """
    def __init__(
        self,
        # MLP-specific parameters
        policy_lm: str,
        hidden_size: int,
        logit_bias: float,
        fluent: bool,
        fluent_top_k: Optional[int],
        # Generation parameters
        max_decoding_length: int,
        eos_token_id: Optional[int]
    ):
        super().__init__()

        assert policy_lm in SUPPORTED_LMS  # TODO: Support more LMs
        model = policy_lm
        self.device = 0  # TODO
        # self.resc = True # Res
        self.resc = False # Res
        self.lmd = 0.2
        self.kl = True # KL-div
        self.eval = False
        print('rlprompt/models/lm_adaptor_model--- device:', self.device)
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
                                            model,
                                            pad_token='<|endoftext|>',
                                            cache_dir='/share/home/wenqingchen/feiyu/RL_debias/Model'
                                            )
        print('AutoTokenizer.from_pretrained--FINISH')
        model_add = GPT2LMHeadModel.from_pretrained(model, 
                                          cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model'
                                          )
        print('AutoModel.from_pretrained--FINISH')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model_add,
                                #   model=model,
                                  device=self.device,
                                #   cache_dir = '/share/home/wenqingchen/feiyu/RL_debias/Model'
                                  
                                  )
        t2 = time.time()
        print('Policy LM loading time:', t2-t1)


        for param in self.generator.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self. fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id

        model_dim = LM_HIDDEN_SIZES[policy_lm]
        self.mlp = _build_one_layer_mlp(in_dim=model_dim,
                                        out_dim=model_dim,
                                        hidden_size=hidden_size).to(self.device)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)
        # if self.eval == True:
        #     self.mlp = 


    def _mlp_forward(self, state: torch.Tensor) -> torch.Tensor:
        # print('state',state)
        mlp_output = self.mlp(state)
        # print('mlp_output',mlp_output)
        if self.resc == True:
            # print('Add residual connections')
            # mlp_output = mlp_output + state #residual connections
            range_ori_mlp_output = torch.max(mlp_output) - torch.min(mlp_output)
            range_state = torch.max(state) - torch.min(state)
            proportion = (range_ori_mlp_output / range_state )
            # print('changed state',self.lmd * proportion * state)
            mlp_output = (1.00 - self.lmd) * mlp_output + self.lmd * proportion * state
            # print('add mlp_output',mlp_output)
            # print('####################SLEEP on lm_adaptor_model.py#########################')
            # time.sleep(15000)
        logits = self.generator.model.lm_head(mlp_output) # distilgpt2        

        if self.fluent:
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        return logits

    def teacher_forcing(
        self,
        source_texts: List[str],
        # BLANK: List[str],
        sample_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        print('lmadaptor def teacher_forcing source_texts',source_texts)
        # print('lmadaptor def teacher_forcing BLANK',BLANK)
        state, past_key_values = self._get_generation_cache(source_texts)

        sample_logits = []
        for i in range(sample_ids.shape[-1]):

            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def sample(
        self,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        print('TEST Sampling mode')
        print('source_texts', source_texts)
        # ['<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>']

        #*************
        # state is logits last layer
        # print('state, past_key_values',state.shape, past_key_values.shape)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        kl_list = []
        # sentence = ['' for _ in source_texts]#1/4
        for i in range(max_new_tokens):
            ori_logits = self.generator.model.lm_head(state)
            logits = self._mlp_forward(state)  # [batch_size, vocab_size] topK already because fluent is true. k=self.fluent_top_k
            logits = logits + self.logit_bias
            if self.kl == True:
                kl = kl_regularization(logits, ori_logits)
                kl_list.append(kl)
                # print('ori_logits',ori_logits,ori_logits.shape)# torch.Size([20, 50257])
                # print('logits',logits,logits.shape)# torch.Size([20, 50257])
                # print('kl',kl)
            # print(logits[:, 4:].min().item(), logits.max().item())

            if top_k is not None:
                sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None:
                sampling_logits = _top_p_logits(logits, p=top_p)
            else:
                sampling_logits = logits

            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]
            # sentence = [stc + tokenstrs for stc, tokenstrs in zip(sentence, token_strs)]#2/4
            # print('token_strs',i, '---',token_strs)#first token in a batch(8/16 * 1)
            # print('sentence',i, '---',sentence)#first token in a batch(8/16 * 1)#


            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))
            # [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)#3/4
            # state, past_key_values = self._get_generation_cache(sentence,
            #                                                     past_key_values)#4/4
        # time.sleep(20000)
        # [batch_size, prompt_length]
        # print('kl_list',kl_list)
        av_kl = sum(kl_list)/len(kl_list)
        print('av_kl',av_kl)
        # print('####################SLEEP on lm_adptor_model.py#########################')
        # time.sleep(200000)
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths,
                      kl_div_20 = av_kl)
        return output

    def greedy_search(self,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")
        
    

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)
            
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())

            actions = logits.argmax(dim=-1)  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        # print('lm_adaptor def _get_generation_cache source_texts:', source_texts)
        token_encoding = (self.generator
                          .tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
                          .to(self.device))
        # print('lm_adaptor def _get_generation_cache token_encoding:', token_encoding)
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        #model here is distilgpt2
        outputs = self.generator.model.transformer(input_ids,
                                                   past_key_values=past_key_values,
                                                   use_cache=True)
        
        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        
        # print('lm_adaptor def _get_generation_cache output:', [self.generator.tokenizer.convert_tokens_to_string([t]) for t in ([self.generator.tokenizer.convert_ids_to_tokens([a])[0] for a in (self._mlp_forward(last_token_hidden_state)).argmax(dim=-1).tolist()])])
        return last_token_hidden_state, past_key_values

    def generate(#
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        # print('do_sample',do_sample)#eval time: all tokens are same--greedy. Training--sample.
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            print('Greedy search')
            return self.greedy_search(source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            print('Sampling')
            return self.sample(source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)


def _build_one_layer_mlp(in_dim, out_dim, hidden_size):
    W1 = nn.Linear(in_dim, hidden_size)
    A1 = nn.ReLU()
    W2 = nn.Linear(hidden_size, out_dim)
    return nn.Sequential(W1, A1, W2)
