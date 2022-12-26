#! /bin/bash
for i in {1..4444};
do
        python3 main.py --env_name $1-v2 --n 200 --m 200 --T 1000 --base_methods 1 --use_meta 1 --meta_model 0| tee -a $1-v2-meps-bies.txt;
        python3 main.py --env_name $1-v2 --n 200 --m 200 --T 1000 --base_methods 1 --use_meta 1 --meta_model 1| tee -a $1-v2-meps-bies-lstm.txt;
done                                                                                                                                                 
