from train_and_test.util import *
import time
a = np.array([(1, 1),
            (2, 2),
            (1, 2)])

start = time.time()
b = geometric_median(a, method='minimize')
end = time.time()
print(end - start)

start = time.time()
c = geometric_median(a, method='weiszfeld')
end = time.time()
print(end - start)

print(b)
print(c)