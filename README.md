# Adversarial-Example-Attack-and-Defense
This repository contains the implementation of the three non-target white box adversarial example attacks and one defense method as countermeasure to those attacks.

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

 
    
## Results
Applied the attack methods and defense uaing MNIST dataset on the model based on [pytorch example model for mnist](https://github.com/pytorch/examples/blob/master/mnist).
### Attacks
#### FGSM
![Test Accuracy](/images/fgsm-attack.png)
![Samples of Adversarial Examples](/images/fgsm-adv.png)
#### I-FGSM
![Test Accuracy](/images/ifgsm-attack.png)
![Samples of Adversarial Examples](/images/ifgsm-adv.png)
#### MI-FGSM
![Test Accuracy](/images/mifgsm-attack.png)
![Samples of Adversarial Examples](/images/mifgsm-adv.png)
### Defense
#### FGSM
![Test Accuracy](/images/defense-fgsm.png)
#### I-FGSM
![Test Accuracy](/images/defense-ifgsm.png)
#### MI-FGSM
![Test Accuracy](/images/defense-mifgsm.png)

