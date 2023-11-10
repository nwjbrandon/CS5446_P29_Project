# Reinforcement Learning for Logistics

This repo contains a gym environment representing a chaotic warehouse, a DQN Agent provided by the library stable-baselines (https://stable-baselines.readthedocs.io) and a simplified implementation of the Intrinsic Curiosity Module (https://pathak22.github.io/noreward-rl/).

<p align="center">
  <img src="Videos/RL.gif" width="300">
</p>

This work was a collaboration between TUM students (Anja Kirschner, Leo
Tappe and Victor Caceres) from different disciplines and myself, under the guidance of the company MaibornWolff GmbH.

For a detailed explanation of the project and its implementation, please visit https://www.di-lab.tum.de/en/past-projects/maibornwolff-multi-agent-reinforcement-learning-for-logistics/.

## Details about this implementation

State:

-Agent Position    
-Agent Status    
-StagingIn (place where the incoming item are)    
-StagingOut (place where the outcoming items are)
-BinStatus (places where the items are located inside the warehouse)
-Timestep     


This implementation uses a Box Space (Continuous Space) instead of a Multidiscrete space, but applies a one-hot encoding to all the variables except the Timestep.

Actions:
LEFT,RIGHT,UP,DOWN + Agent_slots x bin_slots

Particularities    
-No Stay Command    
-Sparse Rewards with only 'completion of an OUTBOUND transaction'    
-Random Transactions (either outbound or inbound sensible transactions are generated when a transaction is completed)    
-Timestep as part of the state    
-The staging-out-area.status details what items are MISSING to complete the transaction    

## Setup
To run the code, create a virtual environment using Anaconda and Python 3.7

1. Clone the repo
```
git clone https://github.com/andresbecker/RL_for_logistics.git
cd RL_for_logistics
```
2. Create a new virtual environment
```
conda create --name conda_rl python=3.7
conda activate conda_rl
```
3. Install the dependencies
```
pip install --upgrade pip wheel
pip install -r requirements.txt
```
4. Install the environment in development mode
```
pip install -e .
```

## Running
Train a DQN on a warehouse just by typing
```
python Train_model.py
```

Your model will be saved to the `models` directory.

## Documentation
You can generate documentation for the entire project by running `make html`
in the `docs` directory. You should now be able to find a file called
`index.html` in `docs/build/html`. Open it with your favorite browser to browse
the documentation.

Note: you need to have `make` installed for this.
