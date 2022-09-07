import torch.multiprocessing as mp 
if __name__  == '__main__':
    for i in range(10):
        p = mp.Process(target=do_something, args=(i, a))
        p.start()
        processes.append(p)
        time.sleep(1)
        
    for p in processes:
        time.sleep(0.1)
        p.join()
    processes.clear()
