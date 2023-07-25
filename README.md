# mlreco3dConfigs
A series of configuration files for mlreco3d package.
The configuration files are divided into various directories designed with a detector configuration as well as a version of mlreco3d in mind.

## Getting set up: GPU resources

To make full use of the ML reco toolkit (*especially* if you want to do any training) you'll need access to GPU(s).
The DeepLearnPhysics group supports can use [only fairly new GPUs](https://hackmd.io/@CuhPVDY3Qregu7G4lr1p7A/SyuW79O4F#Supported-GPUS).  If you don't have your own machine with a supported GPU on it, you have a few options, listed in descending order of desirability:

* SLAC "On Demand" cluster -- Kazu et al. can help arrange
  - Requires authorization (takes ~a week)
  - Lots of high-quality GPUs
  - ~Good availability
* Compute cluster at your institution
  - May be best depending on hardware, availability
  - Depends on local institution support ...
* FNAL's Wilson Cluster has a DUNE allocation
  - Every DUNE collaborator has access
  - Not many GPUs (only 8 compatible with container)
  - Heavily subscribed, so availability is highly variable
  - Maximum slot time is 8h (serious training usually takes longer than this)
  
With access to GPU(s), the configs may ve used independently to suit your workflow or as per the instructions that are a little more prescriptive that are provided in each of the suvdirectories in this repository.




