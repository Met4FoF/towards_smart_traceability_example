# Example accompanying the paper "Towards Smart Traceability"

This is the code for the example given in section 3 of the publication "Towards smart traceability for digital sensors and the industrial Internet of Things".

## What the example is about

The example replays data recorded from a Smart-Up Unit to which an MPC-9250 multi-sensor is connected.
The stream is fetched and processed in an agent-framework.
The pipeline adds input quantization uncertainty, makes the stream equidistant and deconvolves the signal with the (stabilized) inverse sensor transfer behavior.
Finally, the result is plotted in the web-based dashboard of the agent-framework.

## Requirements (Linux)

The following steps assume that you are using a Linux-machine with `git` and `python3`.
Furthermore, we assume that you store git-repos inside `~/git_repos` and python (virtual) environments in `~/python_envs`.
Clone the repo, setup a new virtual Python environment and install the dependencies:

```bash
# clone
cd ~/git_repos
git clone https://github.com/Met4FoF/towards_smart_traceability_example.git
cd towards_smart_traceability_example

# create and activate new python environment
python -m venv ~/python_envs/towards_smart_traceability_example
. ~/python_envs/towards_smart_traceability_example/bin/activate

# install dependencies
pip install -r requirements.txt
```

Download the utilized dataset from Zenodo. The download size is ~170MB, but the extract archive will take up ~4.2GB. 

```bash
cd ~/git_repos/towards_smart_traceability_example/data
wget ...
tar ...
```

## How to run the example

Start in simulation of the Smart-Up Unit:

```bash
cd ~/git_repos/towards_smart_traceability_example/code
. ~/python_envs/towards_smart_traceability_example/bin/activate
python simulate_board.py
```

In a second window, start the agent-network by:

```bash
cd ~/git_repos/towards_smart_traceability_example/code
. ~/python_envs/towards_smart_traceability_example/bin/activate
python agent_network.py
```

You can now open a browser to see the dashboard at [http://localhost:8050].

## Simple modifications of the example

...