#! /bin/bash
for i in {1..20};
do
        python3 main.py --T 100 --t 10 --env_name InvertedDoublePendulum-v2 --m 400 --n 400| tee -a InvertedDoublePendulum-v2-T-100-t-10-m-400-n-400.txt; 
done


