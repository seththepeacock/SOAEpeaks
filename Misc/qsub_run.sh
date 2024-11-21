#!/bin/bash

#PBS -N cnn_tissue_mnist
#PBS -A PAS2038
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=4
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
cd $HOME/osc_classes/PHYSICS_5680_OSU/materials/module_6_cnn_osc
python -u my_cnn_intro.py


