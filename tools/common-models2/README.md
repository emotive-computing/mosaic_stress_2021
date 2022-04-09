# COMMON MODELS: REDESIGNED 
Welcome! This is the repository for the Machine Learning Library used within the Emotive Computing Lab.  We refer to this library as "CM2" below.

## INSTALLATION INSTRUCTIONS
### WINDOWS AND LINUX
1. Make sure you are using python 3.7+.  We suggest using a separate virtual environment for installation. For example, if you are using Anaconda: `conda create --name cm2 python=3.7` will create an environment named *cm2*
2. Activate the new environment: `conda activate cm2`
3. Install the package using pip inside the root folder of the repository: `pip install -e .`.  The `-e` flag performs an editable install, meaning you can update the CM2 code (e.g., using git pull) without having to install again.

### MAC
*Installation has only been tested on Mac M1 Silicon *

#### Mac M1 Silicon Installation Steps:
1. Install Xcode from the App Store or you can download it from the Developer site.
2. Install the Command Line Tools Package on Terminal using: `xcode-select --install`

*The Command Line Tools Package is a small self-contained package available for download separately from Xcode and that allows you to do command line development in macOS.*

### TESTING INSTALLATION
1. Navigate to the samples folder and run one or more of the following, which should complete without errors:
```
python sklearn_test.py
python pytorch_test.py
python tf_test.py
```

## USING THE CU BOULDER CURC CLUSTER
This section is intended for CU Boulder users only, as it pertains to their HPC facilities.

### Getting a CURC account
1. Visit https://www.colorado.edu/rc/ and register for a CURC account
2. Follow the instructions, setup Duo (takes 24 hours)
3. Login to CURC for your first time: `ssh <username>@login.rc.colorado.edu`

### Installation
1. Setup anaconda3
    1. Edit or create a file in your home folder: `~/.condarc`
    2. Add the lines in the code block below, then save:
```
pkgs_dirs:
  - /projects/$USER/.conda_pkgs
envs_dirs:
  - /projects/$USER/software/anaconda/envs
```
2. Setup environment variables and default login behavior
    1. Edit `~/.bashrc` and add the lines below to the bottom of the file.  The home directory quota is small, so these changes enable you to install large packages (e.g., tensorflow, jupyterlab)
    2. Run `source ~/.bashrc` to pull in these changes to your bash environment.  You should notice your command line prompt has changed and now starts with "(base)", which means anaconda3 is working.
```
export PIP_CACHE_DIR=/projects/$USER/.cache/pip
export TMPDIR=/scratch/summit/$USER/tmp
source /curc/sw/anaconda3/latest
```
3. Add your CURC public key to Github:
    1. Login to GitHub, select settings, then SSH and GPG keys.  Click "New SSH Key"
    2. Run `cat ~/.ssh/curc.pub` to print the contents of the file to the terminal screen
    3. Copy the contents of this file to the "Key" window in GitHub (MAKE SURE YOU COPY THE CONTENTS OF `curc.pub` AND NOT `curc`).  Type a meaningful name in the "Name" window like "CU HPC", or whatever you like, then click "Add SSH Key"
4. Run `cd /projects/$USER` ($USER always contains your username)
5. Run `git clone git@github.com:emotive-computing/common-models-redesign.git`
6. Follow the Linux installation instructions at the top of this file

#### Adding GPU Support
1. Add this line to the bottom of your `~/.bashrc` file: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/curc/sw/cuda/11.2/lib64:/curc/sw/cudnn/8.1_for_cuda_11.2/cuda/lib64` (this tells tensorflow where to find CUDA)

#### Adding Jupyter Lab Support
In order to run common-models2 code in Jupyter, you need to setup a Jupyter kernel which uses your anaconda3 "cm2" environment:
1. `conda activate cm2`
2. `pip install ipykernel`
3. `python -m ipykernel install --user --name cm2 --display-name cm2`

##### Adding Jupyter Lab Support for GPU Acceleration
1. Make sure you have completed the sections [Adding GPU Support](#adding-gpu-support) and [Adding Jupyter Lab Support](#adding-jupyter-lab-support) above
2. `conda activate cm2`
3. `pip install jupyterlab`

### Running on CURC
1. Once CM2 is installed on HPC, you can run your code on compute nodes either by submitting it as a batch job or by requesting an interactive compute node and then running your code manually.  See the [CURC documentation on running jobs](https://curc.readthedocs.io/en/latest/running-jobs/running-apps-with-jobs.html) for more details.

#### Running jobs in Jupyter (CPU only)
1. Open a browser and go to: `https://jupyter.rc.colorado.edu/`
2. Select a hardware configuration (job profile) suitable for your job, for example `Summit interactive (1 core, 12 hr, instance access)`
3. Click "Spawn"
4. Open or create a notebook (e.g., `/projects/<username>/my_cm2_notebook.ipynb`) and make sure you select the "cm2" kernel
5. Run code in Jupyter.  When the time for your allocated HPC instance is up (e.g., after 12 hours), you will get a message that Jupyter is unavailable.  Repeat these steps to spin up another instance if needed.

#### Running jobs in Jupyter with GPU Support
1. Make sure you have completed the sections [Adding GPU Support](#adding-gpu-support) and [Adding Jupyter Lab Support](#adding-jupyter-lab-support).  The convenient website `https://jupyter.rc.colorado.edu/` makes it easy to run Jupyter code on HPC, but it only permits CPU jobs.  To run a job with GPU support, you need to request your own GPU instance, run JupyterLab, and setup an ssh tunnel so you can access Jupyter from your computer. Follow the instructions in the sections below.

##### On HPC (login node)
1. `module load slurm/summit`
2. `sinteractive --partition=sgpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --qos=normal --time=12:00:00 --job-name=<pick_any_name> --mail-type=all --mail-user=<your.address@email.com>` (make changes to this line as needed)
    1. This may take a while due to the slurm queue.  You can check on your expected start time using: `squeue --user=$USER --start`
    2. You will receive an email when your GPU node request is serviced. Note the output node name (e.g., sgpu0101)
    3. [**Testing Only**] If you simply want to test out this GPU support process, you can instead request a `sgpu-testing` partition without the --time flag.  This will give you access to a `sgpu-testing` node for 30 minutes and usually works instantly
3. From the sgpu instance: `conda activate cm2`
4. `cd /projects/$USER/` (or cd to wherever you want the jupyter lab's root folder to be)
5. `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`
    1. Note the token (e.g., http://(sgpu0101.rc.int.colorado.edu or 127.0.0.1):8888/?token=de17f49fdb54732ba2eebb132b8641ff40c3adfc324ed41d)

##### On Your Computer
1. `ssh -N -L 8888:sgpu0101:8888 <username>@login.rc.colorado.edu` (make sure you use the name of your sgpu node instead of *sgpu0101*)
2. Open browser to localhost:8888 and copy the token from the jupyterlab execution, or copy the line from the jupyterlab output into your browser that looks like this: `http://127.0.0.1:8888/lab?token=423653e0612a1ea1cfd1602a107a65e0157171643716708b`
3. Open or create a notebook (e.g., `/projects/<username>/my_cm2_notebook.ipynb`) and make sure you select the "cm2" kernel
4. Run code in Jupyter.  When the time for your allocated HPC GPU instance is up (e.g., after 12 hours), you will get a message that Jupyter is unavailable.  Repeat these steps to spin up another instance if needed.
