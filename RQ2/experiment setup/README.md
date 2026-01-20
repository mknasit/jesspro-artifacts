# Github Repository

Jess and its source code can also be obtained from the GitHub repository https://github.com/stschott/jess.

# Compilation of Commit Changes within Java Source Code Repositories (Artifact)

This VM contains the source code of Jess and all other code that is needed to replicate the results of the experiments performed in our paper.

## Required Hardware

We tested the VM in VirtualBox 7.0. Running on a Windows 11 machine with an Intel Core i
7-11850H @ 2.50 GHz and 32GB RAM.

## VM User Credentials

- **Username**: jess
- **Password**: jess 

## Artifact Structure

The artifact is located in the `artifact` directory on the Desktop (`/home/jess/Desktop/artifact`).

- **paper**: Contains the paper that corresponds to this artifact.
- **jess**: Contains the source code to the tool Jess.
- **jess-eval**: Contains the code to perform the experiments from RQ1 and RQ2.
- **jess-eval-aggregator**: Contains the script to aggregate the results for RQ1 and RQ2.
- **build-study & kb-compiler**: Contains the code to perform the experiments from RQ3.
- **build-study-aggregator**: Contains the script to aggregate the results for RQ3.
- **dependency-download-maven-plugin**: Contains our custom dependency download plugin. (used by jess-eval and kb-compiler)
- **src-differ**: Contains code to extract changes made in a specific commit. (used by build-study and kb-compiler) 
- **docker**: Contains the Dockerfile's to build the containers for our experiments.
- **logs**: Contains the output data of the experiments. (will be populated by running the scripts)

## Running the experiments

### Demo Experiments

The full experiments take **multiple days** to run. Therefore we have provided a scaled down version of our experiments which takes only around 10 minutes to finish.
Run `./rq1_rq2_demo.sh` to obtain data from RQ1 and RQ2. Run `./rq3_demo.sh` to obtain data from RQ3. After each script is finished, a text editor window will open, containing the aggregated experiment data in form of a JSON string.

### Full Experiments

To replicate the full experiments run `./rq1_rq2.sh` and `./rq3.sh`. Each script will take **multiple days** to finish.

*Disclaimer*: Our original experiments were performed in November 2023. Since we clone the repositories at their most recent state, the results will be slightly different than the ones reported in the paper. 