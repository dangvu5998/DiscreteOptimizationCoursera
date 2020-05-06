#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model

def solve_by_ortools(node_count, edge_count, edges):
    model = cp_model.CpModel()
    model_nodes = []
    for i in range(node_count):
        model_nodes.append(model.NewIntVar(0, node_count - 1, str(i)))
        model.Add(model_nodes[i] <= i)
    for edge in edges:
        model.Add(model_nodes[edge[0]] != model_nodes[edge[1]])
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    for i in range(node_count):
        print(i, solver.Value(model_nodes[i]))

def init_violations(graph, edges_dict):
    violations = np.zeros(len(graph), dtype=np.int32)
    for i in range(len(violations)):
        for j in edges_dict[i]:
            if graph[i] == graph[j]:
                violations[i] += 1
    return violations

def remove_color(max_color, graph, edges_dict):
    color_to_remove = np.random.randint(max_color)
    new_graph = np.copy(graph)
    for i in range(len(graph)):
        if new_graph[i] > color_to_remove:
            new_graph[i] -= 1
        elif new_graph[i] == color_to_remove:
            new_graph[i] = np.random.randint(max_color)
    return new_graph

def solve(node_count, edges):
    print('Nb of nodes:', node_count)
    graph_solution = np.arange(node_count)
    max_color_solution = node_count
    edges_dict = defaultdict(list)
    for edge in edges:
        edges_dict[edge[0]].append(edge[1])
        edges_dict[edge[1]].append(edge[0])

    violations = np.zeros(node_count, dtype=np.int32)
    for max_color in range(node_count - 1, 0, -1):
        print('Finding solution color:', max_color)
        try_times = 0
        max_try = 100
        solution_found = False
        while try_times < max_try:
            try_times += 1
            max_iter = node_count * 5
            iter = 0
            graph = remove_color(max_color, graph_solution, edges_dict)
            violations = init_violations(graph, edges_dict)
            # print('---gggg----')
            # print(graph)
            # print(violations)
            # print('====')

            total_violations = np.sum(violations)
            while iter < max_iter and total_violations > 0:
                iter += 1
                # find max violation node
                max_violation_nodes = []
                max_violation = 1
                for node in range(len(violations)):
                    if violations[node] > max_violation:
                        max_violation = violations[node]
                        max_violation_nodes.clear()
                        max_violation_nodes.append(node)
                    elif violations[node] == max_violation:
                        max_violation_nodes.append(node)
                max_violation_node = max_violation_nodes[np.random.randint(len(max_violation_nodes))]
                # change color max violation to reduce maximum violation
                new_color = graph[max_violation_node]
                new_color_violation = max_violation
                for color in range(max_color):
                    # skip current color
                    if color == new_color:
                        continue
                    color_violation = 0
                    for node in edges_dict[max_violation_node]:
                        if graph[node] == color:
                            color_violation += 1
                    if color_violation == new_color_violation:
                        if np.random.random() > 0.5:
                            new_color = color
                    elif color_violation < new_color_violation:
                        new_color = color
                        new_color_violation = color_violation
                if new_color == graph[max_violation_node]:
                    continue
                # print('---bbbbb----')
                # print(graph)
                # print(violations)
                # print('====')
                # print('max_node', max_violation_node, new_color, new_color_violation)
                # print('neighbor', edges_dict[max_violation_node])
                for node in edges_dict[max_violation_node]:
                    if graph[node] == graph[max_violation_node]:
                        violations[node] -= 1
                    elif graph[node] == new_color:
                        violations[node] += 1
                violations[max_violation_node] = new_color_violation
                graph[max_violation_node] = new_color
                total_violations = np.sum(violations)
                # print('---gggg----')
                # print(graph)
                # print(violations)
                # print('====')
            if total_violations == 0:
                # print('solution')
                # print(graph)
                # print(violations)
                graph_solution = graph
                max_color_solution = max_color
                solution_found = True
                break
        if solution_found == False:
            break
    solution_str = '{} 0\n'.format(max_color_solution)
    for node in range(len(graph)):
        solution_str += str(graph_solution[node]) + ' '
    return solution_str

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    return solve(node_count, edges)
    # return solve_by_ortools(node_count, edge_count, edges)

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

