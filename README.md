# Adversarial-Example-Attack-and-Defense
This repository contains the PyTorch implementation of the three non-target adversarial example attacks (white box) and one defense method as countermeasure to those attacks.

## Attack
1. Fast Gradient Sign Method(FGSM) - [Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.](https://arxiv.org/abs/1412.6572)
```python
def fgsm_attack(input,epsilon,data_grad):
  pert_out = input + epsilon*data_grad.sign()
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out
```
2. Iterative Fast Gradient Sign Method(I-FGSM) - [A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016.](https://arxiv.org/abs/1607.02533)
```python
def ifgsm_attack(input,epsilon,data_grad):
  iter = 10
  alpha = epsilon/iter
  pert_out = input
  for i in range(iter-1):
    pert_out = pert_out + alpha*data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out
```
3. Momentum Iterative Fast Gradient Sign Method(MI-FGSM) - [Y. Dong et al. Boosting Adversarial Attacks with Momentum. arXiv preprint arXiv:1710.06081, 2018.](https://arxiv.org/abs/1710.06081)
```python
def mifgsm_attack(input,epsilon,data_grad):
  iter=10
  decay_factor=1.0
  pert_out = input
  alpha = epsilon/iter
  g=0
  for i in range(iter-1):
    g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)
    pert_out = pert_out + alpha*torch.sign(g)
    pert_out = torch.clamp(pert_out, 0, 1)
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out
```

## Defense 
1. Defensive Distillation - [Papernot, N., McDaniel, P., Wu, X., Jha, S., and Swami, A. Distillation as a defense to adversarial perturbations against deep neural networks.
arXiv preprint arXiv:1511.04508, 2016b.](https://arxiv.org/abs/1511.04508)

According to the paper, defensive distillation can be done by following procedure:
1) Train a network F on the given training set (X,Y) by setting the temperature of the softmax to T.
2) Compute the scores (after softmax) given by F(X) again and evaluate the scores at temperature T.
3) Train another network F'<sub>T</sub> using softmax at temperature T on the dataset with soft labels (X,F(X)). We refer the model F<sub>T</sub> as the distilled model.
4) Use the distilled network F'<sub>T</sub> with softmax at temperature 1, which is denoted as F'<sub>1</sub> during prediction on test data X<sub>test</sub>(or adversarial examples).

Taken Temperature as 100 for training the NetF and NetF'. 

## Results
- Applied the attack methods and defense using MNIST dataset on the model based on [pytorch example model for mnist](https://github.com/pytorch/examples/blob/master/mnist).
- Here, the attacks are white box as all the knowledge of network hyperparameter setting with the network's achitecture.
- Results tell that FGSM attack reduces the test accuracy from 97.08% to 24.84% with epsilon from 0 to 0.3, whereas I-FGSM with number of iteration as 10 reduces 
test accuracy from 96.92% to 30.54% similar with MI-FGSM with decay factor of 1.0 and iterations of 10, reduction in test accuracy from 97.05% to 30.10% i.e. we can 
say that our attacks to the proposed network was successful and it reduced ~70% of test accuracy in all the three cases for max epsilon of 0.3.
- During the defensive distillation used same network as Net F and for Net F' reduced number of filters to half in each layer to reduce the number of parameters. Temperature of 100 was taken in our case. Results tell that FGSM attack reduces test accuracy from 90.33% to 88.01% with same epsilon range, I-FGSM with iteration of 10 reduced test accuracy 
from 90.80% to 88.16% similar with MI-FGSM with same decay factor of 1.0 and iterations of 10, reduction in test accuracy from 90.26% to 87.97% i.e. we can say that defensive
distillation for the proposed network with temp of 100 was successful and it only reduced ~2% of test accuracy in all the three cases for max epsilon of 0.3.

#### Test Accuracy during attacks
##### FGSM 
![](/images/fgsm-attack.png)
##### I-FGSM 
![](/images/ifgsm-attack.png) 
##### MI-FGSM 
![](/images/mifgsm-attack.png) 
#### Test Accuracy during attack using defensive distillation 
##### FGSM 
![](/images/defense-fgsm.png) 
##### I-FGSM 
![](/images/defense-ifgsm.png) 
##### MI-FGSM 
![](/images/defense-mifgsm.png) 
#### Sample Advesarial Examples
##### FGSM 
![](/images/fgsm-adv.png) 
##### I-FGSM 
![](/images/ifgsm-adv.png) 
##### MI-FGSM 
![](/images/mifgsm-adv.png) 

