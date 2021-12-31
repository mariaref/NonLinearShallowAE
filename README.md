# The dynamics of representation learning in shallow, non-linear autoencoders

The package is written in python and uses the pytorch implementation to ML.
Repository **src** contains the source files.

*train_autoencoder.py* 

Trains (online and off-line) a shallow autoencoder both on a synthetic dataset and on benchmark datasets. 
Includes an implementation of the analytical equations allowing to track the dynamics of online training at all times. 
Included benchmark datasets : Cifar 10 (in gray scale) and FashionMNIST

**Example** of command to train *online*, integrating the analytical equations, on a synthetic dataset:

python3 train_autoencoder.py --D 500 --K 2 --dataset sinusoidal --analytical_updates 1 

**Example** of command to train on finite dataset:

python3 train_autoencoder.py --dataset fmnist

*truncated_vanilla_SGD.py*

Trains an AE using different learning rules for reconstruction: 
- sanger's rule
- vanilla SGD
- truncated version of SGD introduced in the article
Includes the implementation of the analytical equations tracking the dynamics of learning.

**Example** of command to train *online*, integrating the analytical equations, on a synthetic dataset:
python3 truncated_vanilla_SGD.py --K 5 --low_rank 5 --D 500 --analytical 1 
