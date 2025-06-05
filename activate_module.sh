#!/bin/bash

source /etc/profile.d/modules.sh
module load maxwell conda/3.9
. mamba-init
mamba activate gpu_env

