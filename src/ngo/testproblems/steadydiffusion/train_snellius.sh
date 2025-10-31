#!/bin/bash
#SBATCH -J NGO              ### job name
#SBATCH --nodes=1            ### Total number of nodes
#SBATCH --gpus-per-node=1    ### Number of GPUs per node (max. 4)
#SBATCH --ntasks-per-node=1  ### Run one task per GPU!
#SBATCH --cpus-per-task=18   ### (18/16 for a100/h100)
#SBATCH --time=23:59:59      ### wall clock time
#SBATCH --partition=gpu_a100 ### see sinfo -d (gpu_a100,gpu_h100) #SBATCH -o ./%x.%j.out #SBATCH -e ./%x.%j.err
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err

# Load modules for MPI and other parallel libraries 
export MACHINE=snellius

module purge

### loads

module list

export UCX_IB_GPU_DIRECT_RDMA=yes # Allow remote direct memory access from/to G\ PU (req. for MPI-IO)

### set extras
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores

if [ "$SLURM_JOB_PARTITION" = "gpu_a100" ]; then
  #A100 settings
  export GPU_MEMORY_PER_CORE=40000
  export MEMORY_PER_CORE=6826
else
  #H100 settings
  export GPU_MEMORY_PER_CORE=94000
  export MEMORY_PER_CORE=12000
fi

##execute

source ../../../../../../my_venv/bin/activate

srun python train.py
