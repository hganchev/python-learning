import threading
import multiprocessing
import time
import sys

def cpu_func(result, niters):
    '''
    A useless CPU bound function.
    '''
    for i in range(niters):
        result = (result * result * i + 2 * result * i * i + 3) % 10000000
    return result

class CpuThread(threading.Thread):
    def __init__(self, niters):
        super().__init__()
        self.niters = niters
        self.result = 1
    def run(self):
        self.result = cpu_func(self.result, self.niters)

class CpuProcess(multiprocessing.Process):
    def __init__(self, niters):
        super().__init__()
        self.niters = niters
        self.result = 1
    def run(self):
        self.result = cpu_func(self.result, self.niters)

class IoThread(threading.Thread):
    def __init__(self, sleep):
        super().__init__()
        self.sleep = sleep
        self.result = self.sleep
    def run(self):
        time.sleep(self.sleep)

class IoProcess(multiprocessing.Process):
    def __init__(self, sleep):
        super().__init__()
        self.sleep = sleep
        self.result = self.sleep
    def run(self):
        time.sleep(self.sleep)


if __name__ == '__main__':
    cpu_n_iters = 500000
    sleep = 1
    cpu_count = multiprocessing.cpu_count()
    input_params = [
        (CpuThread, cpu_n_iters),
        (CpuProcess, cpu_n_iters),
        (IoThread, sleep),
        (IoProcess, sleep),
    ]
    header = ['nthreads']
    results_list = []
    for thread_class, _ in input_params:
        header.append(thread_class.__name__)
    print(' '.join(header))
    for nthreads in range(1, 2 * cpu_count):
        results = [nthreads]
        for thread_class, work_size in input_params:
            start_time = time.time()
            threads = []
            for i in range(nthreads):
                thread = thread_class(work_size)
                threads.append(thread)
                thread.start()
            for i, thread in enumerate(threads):
                thread.join()
            results.append(time.time() - start_time)
        print(' '.join('{:.6e}'.format(result) for result in results))
        results_list.append(results)

    # plot the results list
    import matplotlib.pyplot as plt
    import numpy as np
    print(results_list)
    plt.figure()
    for i, (thread_class, _) in enumerate(input_params):
        plt.plot(np.array(results_list)[:, 0], np.array(results_list)[:, i + 1], label=thread_class.__name__)
    plt.legend()
    plt.xlabel('Number of threads')
    plt.ylabel('Time (s)')
    # plt.savefig('threading_vs_multiprocessing.png')
    plt.show()


