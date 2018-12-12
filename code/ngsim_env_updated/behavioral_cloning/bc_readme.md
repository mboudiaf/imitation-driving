## Behavioral cloning

### Training a model

The most simple imitation learning algorithm to understand is BC (behavioral cloning). It basically consists in directly mapping states to actions by fitting a feedforward MLP to expert data

To train the default BC model, run :
```
python behavioral_cloning.py
```
A non exhaustive list of parameters you can modify :

- The size of the MLP using.
- The maximum number of epochs for training
- The activations of the network
- The batch size
- The learning rate
 For instance: 
```
python behavioral_cloning.py --mlp_size [128,64,32]
```
will create a network with 3 hidden layers of 128,64 and 32 units respectively. 


The trained model will be saved to models/

### Validating a model

To validate the most last model trained, run :
```
python ./../scripts/imitation/validate_bc.py --model_name 'name_of_your_model' 
```

We also recommend using the argument --n_procs 10 that allows for a parallel roll-out of the trajectories, and a way faster validation. 

### Visualizing the validation metrics

To visualize the validation metrics, you will need to open the notebook ./../visualize.ipynb and follow the instructions. The only thing you should need is to remember the name of the model you used.

