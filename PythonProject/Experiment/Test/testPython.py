import math
import numpy as np
import sys

if __name__ == '__main__':
    print(np.arange(1, 8, 1))
    print(np.linspace(30, 41, 8))
    list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    list1 = np.array(list1)
    print(sys.getsizeof(1000000000000000000000))
    s = '00011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001'
    b = int(s, 2)
    print(b)
    print(format(b, 'b'))
    print(1 << (5 * 8 + 8 - 1 - 0))
    print("-------------------------------------")
    map = {}
    map[1] = 123123123123
    map[2] = 123123
    print(map.keys())
    print(map.values())
    print(map[1] == 123123123123)
