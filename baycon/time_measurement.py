import time

time_to_first_solution = float('inf')
time_to_best_solution = float('inf')
total_time = float('inf')
init_time = 0


def init():
    global init_time, time_to_first_solution, time_to_best_solution, total_time
    init_time = time.process_time()
    time_to_first_solution = float('inf')
    time_to_best_solution = float('inf')
    total_time = float('inf')


def first():
    global time_to_first_solution, init_time
    if time_to_first_solution == float('inf'):
        time_to_first_solution = time.process_time() - init_time
        best()


def best():
    global time_to_best_solution, init_time
    time_to_best_solution = time.process_time() - init_time


def finish():
    global init_time, total_time
    total_time = time.process_time() - init_time
