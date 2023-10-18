# Programming problem

This Python program reproduced and evaluated $\frac{1}{2}$-Tsallis-INF [1] on stochastic MAB problem instances.

## Files

`MAB.py` contains a simple implementation of Multi-Armed Bandit.

`TsallisInf.py` contains the reproduction of $\frac{1}{2}$-Tsallis-INF.

`utils.py` contains some useful tools.

`myinfo.pkl` is used to save and load sample-paths and other information such as the random states.

`*.png` are the curves of $\mathcal{R}(t)$ where the expect regrets are estimated by 10 and 20 epoch respectively.

## Settings

The parameters of the stochastic MAB problem instances are: $T=10000$, $K=8$, and the Bernoulli reward distributions are set to `[0.4,0.5,0.2,0.7,0.1,0.25,0.3,0.1]`. 

The default seed is set to $1$.

The learning rate of OMD is set to $\frac{1}{\sqrt{t}}$.

`cvxpy` and `SCS` solver are used to compute $w_{t}=\nabla\left(\Psi_{t}+\mathcal{I}_{\Delta^{K}}\right)^{*}(-\hat{L}_{t-1})$.

The results in `Evaluation_epoch_20.py` can be reproduced by the following command.

```bash
python .\TsallisInf.py --T 10000 --epoch 20
```

The program can continue the calculation from an interruption by add `--recover` argument if there is a `.pkl` file.

## References

[1] Zimmert J, Seldin Y. An optimal algorithm for stochastic and adversarial bandits[C]//The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019: 467-475.