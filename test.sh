#!/bin/bash
#SBATCH -p sse_llm_projects
#SBATCH -n 1 
#SBATCH -G 1 
#SBATCH -o test.out

python test.py
# nvidia-smi
# python test.py
# nvidia-smi

