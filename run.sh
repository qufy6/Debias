#!/bin/bash
#SBATCH -p sse_llm_projects
#SBATCH -n 1 
#SBATCH -G 1 
#SBATCH -o Experiment/GPT2_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis00.out 

python koala_run.py
# nvidia-smi
# python test.py
# nvidia-smi

