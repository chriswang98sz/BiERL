# BiERL
Implementation of BiERL: A Meta Evolutionary Reinforcement Learning Framework via Bilevel Optimization in PyTorch. The repository works in various MuJoCo and Box2D continuous tasks.

## Requirement
* Python 3.9
* PyTorch 1.12
* Gym 0.15.7
* Mujoco 2.0

# Usage
You can run MEPS+NSRES\main.py or ESAC\main.py in order to run reinforcement learning experiments with BiERL and the three baselines(Vanilla ES, NSR-ES and ESAC). For example, to run the parametric model of BiERL with NSR-ES in Walker2d-v2:

`python MEPS+NSRES\main.py --env_name Walker2d-v2

The hyperparameters can alse be reset: 

`python MEPS+NSRES\main.py --env_name Walker2d-v2 --use_meta 1 --base_methods 3 --meta_model 1 --seed 1 --lr 0.01 --n 40 --k 20`

More initial settings for each methods can be found in main.py.

