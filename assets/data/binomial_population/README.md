Code required to run:

python .\assets\data\binomial_population\main.py --N=<int> --n=<int> --M=<int>
--alpha_h=<float> --r=<float> --beta=<float> --epsilon=<float> --iterations=<int> --increment=<float?>

The following are the definitions of the above parameters:

N: The number of players
n: The number of players contributing a low amount
M: The maximum contribution
alpha_h: The higher contribution
r: The ratio/return on investment
beta: Selection intensity
epsilon: Also selection intensity (see documentation)
iterations: The amount of data to generate. All parameters being incremented
once is defined as one iteration
increment: When a parameter is incremented, it is multiplied by increment

This generates data according to the binomial contribution rule

We save data in the following order:

UID,N,M,i,$\alpha_{i}$,First
Contributor,r,$\beta$,$\epsilon$,$\rho_{C},$p\_{C}$,Process used
