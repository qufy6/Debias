import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.losses import sql_loss_with_sparse_rewards
from rlprompt.utils import utils

import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('rlprompt/modules/sql_module--- sql DEVICE::::::::', device)


class SQLModule(BaseModule):
    # algo_module = make_sql_module(prompt_model, reward, config)
    def __init__(
        self,
        model: BaseModel,#prompt_model policy module
        target_model: Optional[BaseModel],#target module
        reward: Optional[BaseReward],
        sql_loss_impl: str,
        training_mode: str,
        mix_strategy: Optional[str],
        target_update_method: str,
        target_update_steps: Optional[int],
        target_learning_rate: float,
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
    ):
        super().__init__()
        # Initialize self._model and self._reward
        assert target_update_method in ["copy", "polyak"]
        assert not (top_k is not None and top_p < 1.0), \
               "Only one of top_k or top_p should be selected"

        self._model = model#policy
        if target_model is None:
            self._target_model = copy.deepcopy(self._model)
        else:
            self._target_model = target_model
        # for p1, p2 in zip(self._model.parameters(), self._target_model.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         print(False)
        #     print(True) 
        self._reward = reward

        self._sql_loss_impl = sql_loss_impl
        self._training_mode = training_mode
        self._mix_strategy = mix_strategy
        self._forward_modes = _get_forward_modes(training_mode, mix_strategy)
        self._target_update_method = target_update_method
        self._target_update_steps = target_update_steps
        self._target_learning_rate = target_learning_rate
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams

        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

    def _sync_target_model(self) -> None:
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if self._target_update_method == "copy":
            self._target_model.load_state_dict(self._model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._target_update_method == "polyak":
            for param_, param in zip(self._target_model.parameters(),
                                     self._model.parameters()):
                param_.data.copy_((1 - self._target_learning_rate) * param_
                                  + self._target_learning_rate * param)

    def _pre_steps(self, step: int) -> None:
        if self._target_update_method == "polyak":
            self._sync_target_model()
        elif self._target_update_method == "copy" \
                and step % self._target_update_steps == 0:
            self._sync_target_model()

    def forward(self, batch: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        loss_log_list = []
        for mode in self._forward_modes:
            _loss, _loss_log = self._forward(mode=mode, batch=batch)
            loss_list.append(_loss)
            loss_log_list.append(_loss_log)

        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        loss_log = utils.unionize_dicts(loss_log_list)

        return loss, loss_log

    def _forward(
        self,
        mode: ForwardMode,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:
        if mode != ForwardMode.SQL_ON and mode != ForwardMode.INFER:
            # TODO: Enable training modes other than on-policy
            raise NotImplementedError('Only on-policy sampling and greedy '
                                      'inference is supported now')

        if mode == ForwardMode.SQL_ON:
            # print('batch-sql_module:def _forward',batch) #--->['s', 'u', 'a'] 
            # {
            #     'source_texts': ['Puerto rican woman  are beautiful and some have attitudes.', 'Puerto rican woman  are attitudes and some have attitudes.', 'Puerto rican woman  are loud and some have attitudes.'], 
            #     'SAU': ['s', 'u', 'a'], 
            #     'scs_id': ['e3a6b8f4162237e71423009db4d3c5ea', 'd96389aabf1d1616e6eb2e38687476af', '82f43fc0abf0e6b9a80ee390608b9589'], 
            #     'BLANK': ['Most people from Scotland are considered BLANK.', 'Most people from Scotland are considered BLANK.', 'Most people from Scotland are considered BLANK.']}
            # }

            # time.sleep(2000)#--------------------------------------------
            (logits, logits_, output_tokens, output_ids, sequence_lengths, kl_20) = \
                self._decode_sampling(batch=batch)
            # print('output-sql_module:def _forward',output_tokens)#12prompt
            # [
            # ['Following', 'ĠAt', 'Normal', 'Price', 'Following'], 
            # ['Text', 'Price', 'Filter', 'History', 'Game'], 
            # ['Defense', 'Pope', 'Student', 'Menu', 'Fox'], 
            # ['Breaking', 'Job', 'Quantity', 'Tip', 'CHAR']
            # ]
            # print('####################SLEEP on sql_module.py#########################')
            # time.sleep(15000)

        raw_rewards, rewards_log = \
            self.compute_rewards(batch=batch, #use reward
                                  output_tokens=output_tokens,
                                  mode="train")
        
        # print('raw_rewards',raw_rewards)
        # print('ori kl_20',kl_20)
        # KL-Regulation ------------------------------------------------------
        # range_raw_rewards = torch.max(raw_rewards) - torch.min(raw_rewards)
        # range_kl_20 = torch.max(kl_20) - torch.min(kl_20)
        # proportion = (range_raw_rewards / range_kl_20 )/1000
        # beta = -0.03
        # beta = proportion * beta
        # kl_20 = kl_20 * beta
        # shaped_rewards = self._reward_shaping_func(raw_rewards + kl_20)
        shaped_rewards = self._reward_shaping_func(raw_rewards)
        # KL-Regulation ------------------------------------------------------
        # print('scaled kl_20',kl_20)
        # print('plus raw_rewards',raw_rewards + kl_20)
        # print('shaped_rewards.mean()',shaped_rewards)
        # print('plus kl shaped_rewards.mean()',shaped_rewards)
        # print('####################SLEEP on sql_module.py#########################')
        # time.sleep(15000)
        

        sql_loss, sql_loss_log = sql_loss_with_sparse_rewards(
            implementation=self._sql_loss_impl,
            logits=logits,
            logits_=logits_,
            actions=output_ids,
            sampled_actions=None,
            rewards=shaped_rewards,
            sequence_length=sequence_lengths)

        utils.add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode.value}/rewards/")
        utils.add_prefix_to_dict_keys_inplace(
            sql_loss_log, prefix=f"{mode.value}/")
        sql_loss_log = utils.unionize_dicts([
            rewards_log,
            sql_loss_log,
            {
                f"{mode.value}/rewards/raw": raw_rewards.mean(),
                f"{mode.value}/rewards/shaped": shaped_rewards.mean(),
            },
        ])

        return sql_loss, sql_loss_log

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards_tensor, rewards_log = self._reward(
            **batch,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode)

        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor, rewards_log

    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._model.generate(**batch,# generate is in input_conditioned_prompt_model
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    def _decode_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        print('def _decode_sampling: batch',batch)
        # {'source_texts': ['Most people from scotland are considered cupcake.', 'Most people from scotland are considered drunks.', 'Most people from scotland are considered athletic.'], 
        #  'SAU': ['u', 's', 'a'], 
        #  'scs_id': ['1529459367b7a6009ca13aeaf1de7825', '134d7ceceb4c0098221c2040be472790', 'ae54022bd78616225922ec2be4b218f8'], 
        #  'BLANK': ['Most people from Scotland are considered BLANK.', 'Most people from Scotland are considered BLANK.', 'Most people from Scotland are considered BLANK.']
        #  }
        outputs = self._model.generate(**batch,# generate is in input_conditioned_prompt_model
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams)
        # print('outputs IN sql_module.py:', outputs) 
        #---> outp in def generate IN input_conditioned_pro..
        # 4 prompt with length 5

        # self._top_k 256
        # self._top_p 1.0
        # num_beams 1
        # print('outputs',outputs)

        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)
        # print('batch_',batch_)#add sample_logits sample_ids sample_lengths into dict batch

        outputs_ = self._target_model.teacher_forcing(**batch_)
        # print('outputs_ IN sql_module.py:',outputs_)
        

        return (outputs['sample_logits'].contiguous(),
                outputs_['sample_logits'].contiguous(),#only use teacher forcing to get logits
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous(),
                outputs['kl_div_20'].contiguous())


def _get_forward_modes(
    training_mode: str,
    mix_strategy: Optional[str]
) -> List[ForwardMode]:
    if training_mode == "sql-mixed":
        candidate_modes = [
            ForwardMode.SQL_OFF_GT,
            ForwardMode.SQL_ON]

        if mix_strategy == "alternate":
            modes = [candidate_modes[step % len(candidate_modes)]]
        elif mix_strategy == "mix":
            modes = candidate_modes

    else:
        training_mode_map = {"sql-onpolicy": ForwardMode.SQL_ON,
                             "sql-offpolicy": ForwardMode.SQL_OFF_GT}

        modes = [training_mode_map[training_mode]]

    return modes
