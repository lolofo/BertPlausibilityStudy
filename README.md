# BertPlausibilityStudy (NLP)



In this repository we will study the behaviour of the different attention heads in the BERT architecture.
Our goal is to provide a general method to extract from these different heads an attention map which provide a plausible
explanation in a classification task. 
For a sentence s, an attention map will be real vector of the same size as the number of tokens in s. Each component of
the vector will be between 0 and 1 and the vector sum to 1. Each value qualify the role of each token in the decision
process (a value close to 1 is for the tokens which have an important role to play in the decision-making).

Note that to define and to evaluate the plausibility we'll need annotate corpora, this is why we used the following ones.

## Table of Contents
0.[Configuration](#Configuration)
1. [E-SNLI](#ESNLI)
2. [HateXPlain](#HateXPlain)
3. [YelpHat](#YelpHat)

## Configuration

In order to execute the different code you'll need a special python environment.
This environment is available in the `.\requirement\` folder.
To install the environment the command lines are the following :

```commandline
# Specify path to venv
export VENV=path/to/venv
echo $VENV

# Create venv
python -m venv $VENV/bert

# Activate venv
source $VENV/bert/bin/activate

# Replicate on cpu
pip install -r python_env/requirements_bert.cpu.txt --no-cache-dir

# Replicate on gpu
pip install -r python_env/requirements_bert.gpu.txt --no-cache-dir

# Exit venv
deactivate
```

## ESNLI

On the e-snli task we propose a method to combine the different heads of different layers. We select only the layers 4
to 10 thanks to the results on the Shanon entropy (linear decrease of the entropy between these layers).
Thanks to these first results we tried to regularize the network with the entropy of these maps, but it had the effect
to increase the attention on the punctuation and increase the anisotropy of the hidden states.

Then we proposed a second method, based on Lagrangian relaxation, which consists of supervised the entropy of the maps
with the entropy of the human annotation. This method with a good regularization coefficient (0.01) allow us to increase
the plausibility of the map.

## HateXPlain

The datamodule is available but the inference is coming soon

## YelpHat

The datamodule is available but the inference is coming soon
