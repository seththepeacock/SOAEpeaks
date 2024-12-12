#!/bin/bash

#PBS -N pp_k-3_PE_var
#PBS -A PAS2038
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe
#
# Makes sure we use the correct python
module reset
#
# NOTE: next line is for bash only (ask if you use c-shell)
source /fs/ess/PAS2038/PHYSICS_5680_OSU/jupyter/bin/activate
which python
#
cd $HOME/osc_classes/PHYSICS_5680_OSU/materials/SOAEpeaks
python -u get_best_errors.py


