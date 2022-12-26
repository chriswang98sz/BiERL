for i in {1..5};
do
        python3 main.py --env_name $1-v2 --T 1000  --use_meta 1 --meta_model 0| tee -a $1-v2-esac-bies.txt;
	python3 main.py --env_name $1-v2 --T 1000  --use_meta 1 --meta_model 1| tee -a $1-v2-esac-bies-lstm.txt;
        python3 main.py --env_name $1-v2 --T 1000  --use_meta 1 --meta_model 2| tee -a $1-v2-esac-boes.txt;
	python3 main.py --env_name $1-v2 --T 1000  --use_meta 0 --meta_model 2| tee -a $1-v2-esac.txt;
done     

