Code required to run:

python .\assets\data\linear_population\main.py --N=<int> --M=<int>
--r=<float> --beta=<float> --epsilon=<float> --iterations=<int> --increment=<float>

The following are the definitions of the above parameters:

N: The number of players
M: The maximum contribution
r: The ratio/return on investment
beta: Selection intensity
epsilon: Also selection intensity (see documentation)
iterations: The amount of data to generate. All parameters being incremented
once is considered 1 iteration.
increment: When a parameter is incremented, it is multiplied by this amount

This generates data according to the linear contribution rule

We save data in the following order:

UID,N,M,i,$\alpha_{i}$,First
Contributor,r,$\beta$,$\epsilon$,$\rho_{C},$p\_{C}$,Process
