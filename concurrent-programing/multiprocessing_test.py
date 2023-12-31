import multiprocessing
from multiprocessing import Process, Pool , Queue , Pipe
import time

def f_cpu(num):
    """Function that uses CPU"""
    for i in range(5):
        print(f'Function {num}: {i}')
        time.sleep(1)

def main_Process():
    for i in range(5):
        proces = Process(target=f_cpu, args=(i,))
        proces.start()
        proces.join()

def main_Pool():
    # every process uses 1 core if pool is bigger than number of cores then it waits for free core
    with Pool(processes=2) as pool: 
        pool.apply_async(f_cpu, args=(1,))
        pool.apply_async(f_cpu, args=(2,))
        pool.apply_async(f_cpu, args=(3,))
        pool.close()
        pool.join()

def f_queue(q):
    q.put([42, None, 'hello'])

def main_Queue():
    q = Queue()
    p = Process(target=f_queue, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()

def f_pipe(conn):
    conn.send([42, None, 'hello'])
    conn.close()

def main_Pipe():
    parent_conn, child_conn = Pipe()
    p = Process(target=f_pipe, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()

def main(): 
    print('Process') 
    main_Process()
    print('Pool')
    main_Pool()
    print('Queue')
    main_Queue()
    print('Pipe')
    main_Pipe()

if __name__ == '__main__':
    main()