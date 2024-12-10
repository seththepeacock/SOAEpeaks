#!/bin/bash

#PBS -N train_peak_picker
#PBS -A PAS2038
#PBS -l walltime=3:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe
#SBATCH --mail-user=seththepeacock@gmail.com
#
# Makes sure we use the correct python
module reset
#
# NOTE: next line is for bash only (ask if you use c-shell)
source /fs/ess/PAS2038/PHYSICS_5680_OSU/jupyter/bin/activate
which python
#
cd $HOME/osc_classes/PHYSICS_5680_OSU/materials/SOAEpeaks
python -u train_pp.py


