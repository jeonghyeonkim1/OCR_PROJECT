import time
import multiprocessing

start_time = time.time()

def count(name):
    for i in range(1, 5001):
        print(name, " : ", i)

num_list = ['p1', 'p2', 'p3', 'p4']

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)
    pool.map(count, num_list)
    pool.close()
    pool.join()

print("---%s second ---" % (time.time() - start_time))
