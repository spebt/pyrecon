# How to run on the CCR HPC cluster
## Through SSH
  - ```ssh``` to ```vortex.ccr.buffalo.edu```
  - ```git clone``` the repository
  - ```cd image-reconstruction/MLEM-MPI```
  - submit the job with ```sbatch slurm-run.sh```
## Through UB CCR OnDemand
  - Login to the OnDemand
  - Open a command-line shell from the OnDemand dashboard (ccrsoft/legacy should be loaded automaticlly)
    
  ![ondemand-shell](https://github.com/UB-SPEBT/image-reconstruction/assets/48816609/9081346d-a3c5-4f28-bf2e-4d87beabe52a)

  - ```git clone``` the repository
  - ```cd image-reconstruction/MLEM-MPI```
  - submit the job with ```sbatch slurm-run.sh```

## The reconstruction configuration
The configuration is loaded from the ```config.yml``` file. You can change it for your need.
 (but do not push your change to the remote repository, unless necessary) 


