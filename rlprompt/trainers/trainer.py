import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List
import os
import wandb
import json
import click

from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import get_default_train_op, set_random_seed
import pickle
import sys


import time
# def collate_fn(batch):
#     # 这种读了3个BLANK，会跳过没选中的BLANK
#     print('def collate_fn: batch',batch)
#     # [{'source_texts': 'Puerto rican woman  are beautiful and some have attitudes.', 
#     #   'SAU': 's', 
#     #   'scs_id': 'e3a6b8f4162237e71423009db4d3c5ea', 
#     #   'BLANK': 'Puerto Rican woman  are BLANK and some have attitudes.'}, 
      
#     #   {'source_texts': 'Puerto rican woman  are attitudes and some have attitudes.', 
#     #    'SAU': 'u', 
#     #    'scs_id': 'd96389aabf1d1616e6eb2e38687476af', 
#     #    'BLANK': 'Most people from Scotland are considered BLANK.'}, 
       
#     #    {'source_texts': 'Puerto rican woman  are loud and some have attitudes.', 
#     #     'SAU': 'a', 
#     #     'scs_id': '82f43fc0abf0e6b9a80ee390608b9589', 
#     #     'BLANK': 'The Sudanese man was very BLANK.'}]
#     selected_source_texts = [sample['source_texts'] for sample in batch[:3]]
#     selected_SAU = [sample['SAU'] for sample in batch[:3]]
#     selected_scs_id = [sample['scs_id'] for sample in batch[:3]]
#     selected_BLANK = [sample['BLANK'] for sample in batch[:1]]
#     print('def collate_fn 3',selected_source_texts, selected_SAU, selected_scs_id, selected_BLANK)

#     return {
#         'source_texts': selected_source_texts,
#         'SAU': selected_SAU,
#         'scs_id': selected_scs_id,
#         'BLANK': selected_BLANK
#     }

def collate_fn(batch):
    #Here batch is:
    # [{'source_texts': 'Puerto rican woman  are beautiful and some have attitudes.', 
    #   'SAU': 's', 
    #   'scs_id': 'e3a6b8f4162237e71423009db4d3c5ea', 
    #   'BLANK': 'Puerto Rican woman  are BLANK and some have attitudes.'}, 
      
    #   {'source_texts': 'Puerto rican woman  are attitudes and some have attitudes.', 
    #    'SAU': 'u', 
    #    'scs_id': 'd96389aabf1d1616e6eb2e38687476af', 
    #    'BLANK': 'Most people from Scotland are considered BLANK.'}, 
       
    #    {'source_texts': 'Puerto rican woman  are loud and some have attitudes.', 
    #     'SAU': 'a', 
    #     'scs_id': '82f43fc0abf0e6b9a80ee390608b9589', 
    #     'BLANK': 'The Sudanese man was very BLANK.'}]
    selected_source_texts = []
    selected_SAU = []
    selected_scs_id = []
    selected_BLANK = []

    for idx, sample in enumerate(batch):
        if idx < 3:
            selected_source_texts.append(sample['source_texts'])
            selected_SAU.append(sample['SAU'])
            selected_scs_id.append(sample['scs_id'])
        if idx == 0:
            selected_BLANK.append(sample['BLANK'])

    return {
        'source_texts': selected_source_texts,
        'SAU': selected_SAU,
        'scs_id': selected_scs_id,
        'BLANK': selected_BLANK * 3
    }

class Trainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
        self,
        module: BaseModule,#including model:policy module, target_model: target
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str]
    ):
        assert do_eval == False or eval_dataset is not None, \
            "Need to have eval_dataset if do_eval is True"
        self.module = module#policy module

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps

        self.train_op = get_default_train_op(self.module._model,
                                             learning_rate,
                                             gradient_clip,
                                             gradient_clip_norm)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        self.run_name = run_name

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))

    

    def _get_train_dataloader(self) -> DataLoader:
        print(len(self.train_dataset))#----->3
        print('self.train_batch_size,',self.train_batch_size)#---->3
        # time.sleep(20000)
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,#batch_size = 3
                          drop_last=self.train_drop_last,
                        #   collate_fn=collate_fn
                          )

    # @torch.no_grad
    def _train_step(
        self,
        step: int,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.module.train()
        model._pre_steps(step)#def _pre_steps sql_module line 91

        loss, batch_log = model(batch)

        loss.backward()
        self.train_op()

        return batch_log

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None: 
            run_name = self.run_name
        if config is not None: 
            config = eval(str(config))
        if report_to_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
            wandb.watch(self.module, log=None)

        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        train_dataloader = self._get_train_dataloader()
        # print('train_dataloader=============================\n', train_dataloader)
        # print('train_dataloader=============================\n', len(train_dataloader))
        # print('train_dataloader=============================\n', train_dataloader[0])

        # Determine whether to train by epoch or steps
        if self.max_train_steps < 0:
            total_train_epochs = self.num_train_epochs#1
        else:
            num_batches_per_epoch = len(train_dataloader)#3000
            print('num_batches_per_epoch',num_batches_per_epoch)
            #if u wanna train 12000 steps, but u only have 3000 train batches, it will go 4 epoches.
            total_train_epochs = \
                (self.max_train_steps // num_batches_per_epoch
                 + int(self.max_train_steps % num_batches_per_epoch > 0))

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0

        total_steps = 0
        print('total_train_epochs',total_train_epochs)
        for epoch in range(total_train_epochs):
            #-----------------Eval Block Begining-----------------------
            # print('Start media ALL Eval--')
            # output_save_path = \
            #     os.path.join(eval_save_dir,
            #                     f'outputs.step.{total_steps}.json')
            # eval_log = self.evaluate(output_save_path=output_save_path)#eval batch_size clusters
            # print('Finish media ALL Eval--')
            # sys.exit()
            #-----------------Eval Block End-----------------------
            for step, batch in enumerate(train_dataloader):#1 cluster
                # if step == 2:
                #     print('####################SLEEP on trainer#########################')
                #     time.sleep(15000)
                print('Start Train--',step)
                batch_log = self._train_step(step, batch)
                print('Finish Train--',step)
                if report_to_wandb:
                    wandb.log(batch_log)
                total_steps += 1

                if self.do_eval and eval_by_steps \
                        and total_steps % self.eval_steps == 0:
                    print('Start media Eval--')
                    output_save_path = \
                        os.path.join(eval_save_dir,
                                     f'outputs.step.{total_steps}.json')
                    eval_log = self.evaluate(output_save_path=output_save_path)#eval batch_size clusters
                    print('Finish media Eval--')
                    if report_to_wandb:
                        wandb.log(eval_log)

                if self.do_save and save_by_steps \
                        and total_steps % self.save_steps == 0:
                    torch.save({"steps": total_steps,
                                "model_state_dict": self.module.state_dict()},
                               os.path.join(ckpt_save_dir,
                                            f"ckpt.step.{total_steps}.pth"))

                if total_steps == self.max_train_steps:
                    break

            if self.do_eval and not eval_by_steps:
                output_save_path = os.path.join(eval_save_dir,
                                                f'outputs.epoch.{epoch+1}.json')
                eval_log = self.evaluate(output_save_path=output_save_path)
                wandb.log(eval_log)

            if self.do_save and not save_by_steps:
                torch.save({"steps": total_steps,
                            "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir,
                                        f"ckpt.epoch.{epoch+1}.pth"))

    # def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
    #     return DataLoader(eval_dataset,
    #                       batch_size=self.eval_batch_size)
    
    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        print('Eval test, self.eval_batch_size:',self.eval_batch_size)#---->
        # time.sleep(20000)
        return DataLoader(eval_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.eval_batch_size,#batch_size = 3
                          )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        output_save_path: Optional[str] = None,
        compute_scores: bool = True
    ) -> Dict[str, np.number]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        print('Eval test, len(eval_dataset):',len(eval_dataset))#----->60
        eval_dataloader = self._get_eval_dataloader(eval_dataset)#Like def _get_train_dataloader
        print('Eval test, len(eval_dataloader):',len(eval_dataloader))#----->1

        model = self.module.eval()
        hypos = []
        scores: List[List[str]] = []
        test_step = 0

        for batch in eval_dataloader:#20 in total, 20 is length of eval_dataset
            test_step += 1
            # if test_step == 3:
            #     print('####################SLEEP on trainer#########################')
            #     time.sleep(15000)
            print(test_step, 'batch', batch)
            infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
            # print('model',model)#InputConditionedPromptModel-LMAdaptorModel-generate
            infer_outputs = model.infer(batch)
            print(test_step, 'infer_outputs', infer_outputs)
            hypos += infer_outputs['sample_tokens']


            score, score_log = model.compute_rewards(
                batch=batch,
                output_tokens=infer_outputs['sample_tokens'])
            print(test_step, 'this score', score)    
            scores += score.detach().tolist()
        print('Eval test, scores:',scores)#average eval score

        

        # score = score.mean().item()
        # score = scores.mean().item()
        score = sum(scores) / len(scores)
        print('Eval test, score:',score)
        if output_save_path is not None:
            json.dump({'output_tokens': hypos,
                       'scores': scores,
                       'EvalTest-AVGscore:': score},
                      open(output_save_path, 'w'))
        
        # time.sleep(2000)
        utils.add_prefix_to_dict_keys_inplace(
            score_log,#20 for 1 batch
            prefix=f"eval/rewards/")
        
        # print('Finish Eval')

        # print('####################SLEEP on trainer#########################')
        # time.sleep(15000)
        
        return utils.unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"eval/score": score,
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ])
