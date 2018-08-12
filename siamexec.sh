#!/bin/bash
sbatch --gres=gpu:1 --partition=zirconium --wrap="./siamenv.sh"
