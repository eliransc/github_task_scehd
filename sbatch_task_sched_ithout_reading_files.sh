#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 10000
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/github_task_scehd/code_sub_2.py
