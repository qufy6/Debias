import torch
from typing import Optional, List, Union, Any, Dict
from .base_model import BaseModel
import time

class InputConditionedPromptModel(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        prompt_length: int,
        source_train_reps: int,
        source_infer_reps: int
    ):
        super().__init__()
        self._model = model
        self.prompt_length = prompt_length
        self.source_train_reps = source_train_reps
        self.source_infer_reps = source_infer_reps

    def _do_source_reps(
        self, 
        source_texts: List[str], 
        num_reps: int
    ) -> List[str]:
        source_reps = []
        for text in source_texts: 
            for _ in range(num_reps): 
                source_reps.append(text)
        return source_reps

    def generate(#from /models/lm_adaptor_model: line 257
        self,
        source_texts: List[str],
        BLANK: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        num_beams: Optional[int],
        max_new_tokens: Optional[int] = None,
        infer: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if max_new_tokens is None: 
            max_new_tokens = self.prompt_length
        if infer: 
            num_reps = self.source_infer_reps
        else: 
            num_reps = self.source_train_reps
        
        # print('Input_condi: BLANK',BLANK)
        # Input_condi: BLANK 
        # [
        # 'Most people from Scotland are considered BLANK.', 
        # 'Most people from Scotland are considered BLANK.', 
        # 'Most people from Scotland are considered BLANK.'
        # ]
        # print('Input_condi: source_texts',source_texts)
        #['Puerto rican woman  are beautiful and some have attitudes.', 
        # 'Puerto rican woman  are attitudes and some have attitudes.', 
        # 'Puerto rican woman  are loud and some have attitudes.']

        # source_reps = self._do_source_reps(source_texts, num_reps)
        # print('Input_condi: repeated source_texts',source_reps)
        # ['Puerto rican woman  are beautiful and some have attitudes.', 
        # 'Puerto rican woman  are beautiful and some have attitudes.', 
        # 'Puerto rican woman  are beautiful and some have attitudes.', 
        # 'Puerto rican woman  are beautiful and some have attitudes.', 
        # 'Puerto rican woman  are attitudes and some have attitudes.', 
        # 'Puerto rican woman  are attitudes and some have attitudes.', 
        # 'Puerto rican woman  are attitudes and some have attitudes.', 
        # 'Puerto rican woman  are attitudes and some have attitudes.', 
        # 'Puerto rican woman  are loud and some have attitudes.', 
        # 'Puerto rican woman  are loud and some have attitudes.', 
        # 'Puerto rican woman  are loud and some have attitudes.', 
        # 'Puerto rican woman  are loud and some have attitudes.']

        source_reps = self._do_source_reps([BLANK[0]], num_reps)
        print('Input_condi generate input:', source_reps)
        # input_condi generate input: ['Puerto Rican woman  are BLANK and some have attitudes.', 'Puerto Rican woman  are BLANK and some have attitudes.', 'Puerto Rican woman  are BLANK and some have attitudes.', 'Puerto Rican woman  are BLANK and some have attitudes.']


        # time.sleep(909999)
        outp = self._model.generate(source_texts=source_reps,#policy -> lm_adaptor
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    num_beams=num_beams,
                                    max_new_tokens=max_new_tokens,
                                    **kwargs)
        print('Input_condi: outp',outp['sample_tokens'])
        # Input_condi: outp {'sample_tokens': [['Game', 'Image', 'ĠSAN', 'Fire', 'WASHINGTON'], ['ĠWe', 'Players', 'Filter', 'ĠFormer', 'ĠA'], ['Weapon', 'Distance', 'Student', 'Temperature', 'Server'], ['Senator', 'Login', 'Profile', 'Share', 'Settings']], 
        # 'sample_logits': tensor(
        # [
        # 4个：
        # [[-2.4955e-05, -5.9402e-05, -1.0667e-04,  ..., -5.3233e-05, -9.4653e-06, -4.7254e-05],
        #  [-2.4958e-05, -5.9409e-05, -1.0665e-04,  ..., -5.3224e-05, -9.4372e-06, -4.7244e-05],
        #  [-2.4963e-05, -5.9403e-05, -1.0664e-04,  ..., -5.3242e-05, -9.4477e-06, -4.7247e-05],
        #  [-2.4961e-05, -5.9415e-05, -1.0664e-04,  ..., -5.3217e-05, -9.4261e-06, -4.7230e-05],
        #  [-2.4953e-05, -5.9399e-05, -1.0664e-04,  ..., -5.3218e-05, -9.4305e-06, -4.7254e-05]],

        # ], device='cuda:0', grad_fn=<CatBackward0>), 
        # 'sample_ids': tensor([
        # [ 8777,  5159, 37376, 13543, 21793],
        # [  775, 24860, 22417, 14466,   317],
        # [27632, 45767, 38778, 42492, 10697],
        # [29774, 47790, 37046, 11649, 26232]], device='cuda:0'), 
        # 
        # 'sample_lengths': tensor([5, 5, 5, 5], device='cuda:0')}

        # print('####################SLEEP on input_conditioned_prompt_model.py#########################')
        # time.sleep(15000)
        return outp

    def teacher_forcing(#from /models/lm_adaptor_model: line 89
        self,
        source_texts: List[str],
        BLANK: List[str],
        sample_ids: torch.LongTensor,
        **kwargs
    ) -> Dict[str, Any]:
        print('source_texts in input_c def teacher',source_texts)
        print('BLANK in input_c def teacher',BLANK)
        source_texts = [BLANK[0]]
        source_reps = self._do_source_reps(source_texts, self.source_train_reps) #original->source text
        print('source_reps',source_reps)
        return self._model.teacher_forcing(source_texts=source_reps,
                                           sample_ids=sample_ids,
                                           **kwargs)
        # source_reps = self._do_source_reps(BLANK, self.source_train_reps)
        # print('source_reps',source_reps)
        # return self._model.teacher_forcing(BLANK=source_reps,
        #                                    sample_ids=sample_ids,
        #                                    **kwargs)

