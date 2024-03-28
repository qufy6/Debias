#!/bin/bash
#SBATCH -p sse_llm_projects
#SBATCH -n 1 
#SBATCH -G 1 
#SBATCH -o Evaluation/_EVAL_Vicuna_MBPS_Sentence_TopKis256_KLisNo-ResCisNO_ALPHAis10.out

python eval_data.py
# nvidia-smi
# python test.py
# nvidia-smi

