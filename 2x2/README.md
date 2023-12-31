
# 2x2 Configuration files

These are the set of configuration files designed to train and validate [lartpc_mlreco3d](https://github.com/DeepLearnPhysics/lartpc_mlreco3d).
The configuration files without 'val' in the filename are for training, whereas the ones that do contain 'val' in the filename are meant for validation. These are to be run when you already have a set of weights that may be generated by running the training configurations.

A more detailed tutorial for the lartpc_mlreco3d package may be found [here](https://www.deeplearnphysics.org/lartpc_mlreco3d_tutorials/index.html). Although this tutorial may be a little outdated, the concepts expressed therein are still valid, even if the technical details of the configuration have changed.

The configuration files provided in this directory are meant for training and validating the various stages of the reconstruction chain independently. 
They are designed for version 2.8.4 of lartpc_mlreco3d and use simulation files generated by Kazu, located on the S3DF cluster at SLAC. 

The simulation files may be found in the directory:

```bash
/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/
```

The file that the configuration files train on is located at

```bash
/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/train.root
```

The file for validation are located at 

```bash
/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root
```

The shell scripts are meant to submit batch jobs to SLURM for training and validation.
The shell scripts assume that your account has access to the neutrino account for running jobs and therefore can access the ampere partition. 

# Quickstart

This quickstart guide is meant to get you started in running these configurations and as such is quite prescriptive. These files can definitely be used in ways that integrate into your current workflow. However, they may need to be edited for that to work and this quickstart guide does not go into how to do that since everyone's workflow may be different.

Furthermore, this guide assumes that you already have access to both an S3DF account and that your account is authorized to use the neutrino account when submitting batch jobs.

## Accessing S3DF

For running these jobs, you can log into S3DF using ssh:

```bash
ssh <Username>@s3dflogin.slac.stanford.edu
ssh neutrino
```

where `<Username>` in the previous command is meant to be replaced with your S3DF username. 

## Setting up a working area 

It is a good idea to have these files running from a working area rather than from your home directory. 
A good place to set up your working are is in 

```bash
/sdf/group/neutrino/$USER
```

The shell scripts require environmental variable `$APP_DIR` that points to your working area. If you already have one that you would like to use, then set the variable `$APP_DIR` to point to that directory. Otherwise, you can create the recommended area using 

```bash
mkdir -p /sdf/group/neutrino/$USER
export APP_DIR=/sdf/group/neutrino/$USER
cd $APP_DIR
```

## Getting the scripts

Next, clone this repository under your `APP_DIR`. If you have git set up with ssh and want to use the ssh version of this repository, it can be cloned with 

```bash
git clone git@github.com:OrghoN/mlreco3dConfigs.git
```

If you dont have ssh set up with github or just want to use the https URL, you can clone it with 

```bash
git clone https://github.com/OrghoN/mlreco3dConfigs.git
```

## lartpc_mlreco3d

Next, clone lartpc_mlreco3d under your `APP_DIR`. If you already have this you can export the environment variable `$MLRECO3D_PATH` to that installation directory.

Otherwise we can get it with 

```bash
git clone git@github.com:DeepLearnPhysics/lartpc_mlreco3d.git
```

for the ssh version of this directory and 

```bash
git clone https://github.com/DeepLearnPhysics/lartpc_mlreco3d.git
```

for the https version. After cloning lartpc_mlreco3d, checkout the tag `v2.8.4` using `git checkout`:.

```bash
cd $MLRECO3D_PATH
git checkout -v 2.8.4
```

We can point the environmental variable `$MLRECO3D_PATH` to this repo by running

```bash
export MLRECO3D_PATH=$APP_DIR/lartpc_mlreco3d
```

These environment variables need to be set every time a job is run with these scripts. To not have to type in the export commandsd each time, you can put those export lines in yuour `.bashrc` or equivalent. You can also verify your paths are set correctly using `env`:

```bash
env | grep APP_DIR
env | grep MLRECO3D_PATH
```

## Submitting jobs

We can now submit SLURM jobs by pointing the `sbatch` command to a batch script:

```bash
cd $APP_DIR/mlreco3dConfigs/2x2
sbatch <filename>.sh
```

Where the `<filename>` is to be replaced with the file you actually want to run, e.g. `train_uresnet_ppn.sh`. 

#### SLURM job management

The status of running jobs can be checked with

```bash
squeue --me
```

Jobs can be cancelled manually with

```bash
scancel <JobId>
```

Where `<JobId>` is to be replaced with the actual id that you can get from running the command to get job status.

