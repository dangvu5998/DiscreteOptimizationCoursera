#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_by_dp(item_count, capacity, items):
    dp = np.zeros((item_count + 1, capacity + 1), dtype=np.int32)
    def generage_capacities(items, capacity):
        weights = set()
        weights.add(0)
        for item in items:
            for weight in list(weights):
                weight = weight + item.weight
                if weight < capacity:
                    weights.add(weight)
        return weights

    weights = list(generage_capacities(items, capacity))
    for i in range(1, item_count + 1):
        for j in weights:
            weight, value = items[i - 1].weight, items[i - 1].value
            if j >= weight:
                dp[i, j] = max(
                    dp[i - 1, j], 
                    dp[i - 1, j - weight] + value)
            else:
                dp[i, j] = dp[i - 1, j]
    for i in range(capacity + 1):
        if dp[item_count, i] > dp[item_count, capacity]:
            capacity = i
    result = '{} 1\n'.format(dp[item_count, capacity])
    choose_items = ['0'] * item_count
    while item_count > 0 and capacity > 0:
        if dp[item_count, capacity] == dp[item_count - 1, capacity]:
            item_count -= 1
        else:
            choose_items[item_count - 1] = '1'
            capacity -= items[item_count - 1].weight
            item_count -= 1
    result += ' '.join(choose_items)
    return result

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    if item_count*capacity < 30000000:
        return solve_by_dp(item_count, capacity, items)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

