Code required to run:

python .\assets\data\binomial_population\main.py --N=<int> --n=<int> --M=<int>
--alpha_h=<float> --r=<float> --beta=<float> --epsilon=<float> --lim=<bool>
--max=<int>

The following are the definitions of the above parameters:

N: The number of players
n: The number of players contributing a low amount
M: The maximum contribution
alpha_h: The higher contribution
r: The ratio/return on investment
beta: Selection intensity
epsilon: Also selection intensity (see documentation)
lim: Whether to impose a limit on the amount of data generated
max: The amount of data to generate (only relavent if lim is True)
inc: The ratio at which parameters are incremented at each time step

This generates data according to the binomial contribution rule

We save data in the following order:

UID,N,M,i,$\alpha_{i}$,First
Contributor,r,$\beta$,$\epsilon$,$\rho_{C},$p\_{C}$,Process used
