#!/bin/bash
# Latest recorded DAOD experiment launcher.

#SBATCH --job-name=DAOD
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36

#SBATCH -o %j.out
#SBATCH -e %j.err

module load matlab2024b

matlab <run1.m
