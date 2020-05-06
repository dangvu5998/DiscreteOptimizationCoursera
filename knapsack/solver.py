#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import copy
import random
import numpy as np
from collections import namedtuple
from ortools.algorithms import pywrapknapsack_solver
Item = namedtuple("Item", ['index', 'value', 'weight'])

class Result:
    value = 0
    chosen_items = None

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
    chosen_items = ['0'] * item_count
    while item_count > 0 and capacity > 0:
        if dp[item_count, capacity] == dp[item_count - 1, capacity]:
            item_count -= 1
        else:
            chosen_items[item_count - 1] = '1'
            capacity -= items[item_count - 1].weight
            item_count -= 1
    result += ' '.join(chosen_items)
    return result

def solve_by_greedy(item_count, capacity, items):
    items = sorted(items, key=lambda item: item.value/item.weight)
    limit_time = 240
    depth_search = 5
    width_search = 6
    start_time = time.time()
    result = Result()
    result.value = 0
    result.chosen_items = ['0'] * item_count

    def LDS(result, curr_value, curr_weight, curr_chosen_items, skip_items=None):
        now = time.time()
        if now - start_time > limit_time:
            return
        if skip_items is None:
            skip_items = set()
        new_chosen_items = []
        for i in range(item_count):
            if i in skip_items or curr_chosen_items[items[i].index] == '1':
                continue
            if curr_weight + items[i].weight <= capacity:
                curr_weight += items[i].weight
                curr_value += items[i].value
                curr_chosen_items[items[i].index] = '1'
                new_chosen_items.append(i)
        if curr_value > result.value:
            result.value = curr_value
            result.chosen_items = copy.copy(curr_chosen_items)

        if len(skip_items) > depth_search or len(new_chosen_items) == 0:
            return
        for i in range(len(new_chosen_items) * width_search):
            k = min(len(new_chosen_items), random.randint(1, width_search))
            curr_skip_items = random.sample(new_chosen_items,  k)
            for curr_skip_item in curr_skip_items:
                item = items[curr_skip_item]
                skip_items.add(curr_skip_item)
                if curr_chosen_items[item.index] == '1':
                    curr_chosen_items[item.index] = '0'
                    curr_value -= item.value
                    curr_weight -= item.weight
            LDS(result, curr_value, curr_weight, copy.copy(curr_chosen_items), copy.copy(skip_items)) 
            for curr_skip_item in curr_skip_items:
                skip_items.remove(curr_skip_item)
                if curr_chosen_items[item.index] == '0':
                    curr_chosen_items[item.index] = '1'
                    curr_value += item.value
                    curr_weight += item.weight
    LDS(result, 0, 0, ['0'] * item_count)
    result_str = '{} 0\n'.format(result.value)
    result_str += ' '.join(result.chosen_items)
    return result_str
    
def solve_by_ortools(item_count, capacity, items):
    solver = pywrapknapsack_solver.KnapsackSolver(
             pywrapknapsack_solver.KnapsackSolver
             .KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
             'Solver')
    profits = []
    weights = []
    for item in items:
        profits.append(item.value)
        weights.append(item.weight)
    solver.set_time_limit(10)
    solver.Init(profits, [weights], [capacity])
    profit = solver.Solve()
    result_str = '{} 0\n'.format(profit)
    result_str += ' '.join([str(int(solver.BestSolutionContains(i))) for i in range(item_count)])
    return result_str

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
    if item_count*capacity < 3000000:
        return solve_by_dp(item_count, capacity, items)
    # return solve_by_greedy(item_count, capacity, items)
    return solve_by_ortools(item_count, capacity, items)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

