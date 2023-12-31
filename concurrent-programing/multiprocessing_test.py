import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue
import time

def function(num):
    """Function that uses CPU"""
    for i in range(5):
        print(f'Function {num}: {i}')
        time.sleep(1)

def main_Process():
    for i in range(5):
        proces = multiprocessing.Process(target=function, args=(i,))
        proces.start()
        proces.join()

def main_Pool():
    # every process uses 1 core if pool is bigger than number of cores then it waits for free core
    with Pool(processes=2) as pool: 
        pool.apply_async(function, args=(1,))
        pool.apply_async(function, args=(2,))
        pool.apply_async(function, args=(3,))
        pool.close()
        pool.join()

def f(q):
    q.put([42, None, 'hello'])

def main_Queue():
    q = Queue()
    p = multiprocessing.Process(target=f, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()

def main(): 
    print('Process') 
    main_Process()
    print('Pool')
    main_Pool()
    print('Queue')
    main_Queue()

if __name__ == '__main__':
    main()