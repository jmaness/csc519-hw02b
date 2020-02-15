## Install prerequisites
Before running any training or gradient checks, 
you must install some prerequisite Python packages.

```
cd assignment
pip install -r requirements.txt 
```


## Check gradient implementation
To verify the gradient implementation, run the following:

```
cd assignment/experiment
python mlp.py --gradient
```

## Train
To train an MLP, run the following:

```
cd assignment/experiment
python mlp.py --train
```