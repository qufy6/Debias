import os
import hydra
from omegaconf import DictConfig, OmegaConf
import time

from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
import dataloader

from koala_helper import (#PromptedClassificationRewardConfig,
                         #FewShotClassificationDatasetConfig,
                        #  make_prompted_classification_reward,
                         make_stereoset_id_dataset)
# from koala_reward import PromptDebiasReward
from koala_rewardfluent import PromptDebiasReward

#------------------------------- general ------------------------------------------
# from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
#                             make_lm_adaptor_model, make_single_prompt_model)
# config_list = [ LMAdaptorModelConfig,
#                 SinglePromptModelConfig, 
#                 SQLModuleConfig, TrainerConfig]
#------------------------------- general ------------------------------------------



#------------------------------- conditional ------------------------------------------
from rlprompt.models import (LMAdaptorModelConfig, InputConditionedPromptModelConfig,
                             make_lm_adaptor_model, make_input_conditioned_prompt_model)
config_list = [ LMAdaptorModelConfig,
                InputConditionedPromptModelConfig,
                SQLModuleConfig, TrainerConfig]
#------------------------------- conditional ------------------------------------------

cs = compose_hydra_config_store('base_koala', config_list)


@hydra.main(version_base=None, config_path="./", config_name="config_koala_rlprompt")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    # input_file = os.path.join("./data", "test.json")  # Eval
    input_file = os.path.join("./data", "dev.json")  # Train
    # input_file = os.path.join("/home/zhichao/qufeiyu/vi", "test.json")
    # input_file = os.path.join("/home/zhichao/qufeiyu/vi", "dev.json")
    # train_id, val_id, test_id = make_stereoset_id_dataset(input_file)
    # print('input_file', input_file)
    
    

    train_dataset, val_dataset, test_dataset = make_stereoset_id_dataset(input_file)
    # print('*******************',train_dataset[0])
    # print('Train Size:', len(train_dataset))#train_dataset is a list of 10 clusters
    # print('11111111111111111111111111111111111111111111:', len(train_dataset[:2]['S_id']))
    # print('Examples:', train_dataset[:5])
    # print('Val Size', len(val_dataset))
    # print('Examples:', val_dataset[:5])
    # print('A')
    policy_model = make_lm_adaptor_model(config)
    # print('B')

    #------------------------------- general ------------------------------------------
    # prompt_model = make_single_prompt_model(policy_model, config)


    #------------------------------- conditional ------------------------------------------
    prompt_model = make_input_conditioned_prompt_model(policy_model, config)#Conditional


    reward = PromptDebiasReward(config)
    algo_module = make_sql_module(prompt_model, reward, config)
    
    
    # Hack for few-shot classification - Each batch contains all examples
    #------------------------------- general ------------------------------------------
    # config.train_batch_size = len(train_dataset)
    # config.eval_batch_size = len(val_dataset)
    #-------------------------------------------------------------------------
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
