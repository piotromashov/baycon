import numpy as np

def generate_random_alternatives(dataset, n):
    # return dataset[random.choices(range(len(dataset)), k=n)
    #getting the last n as "random"
    return dataset[len(dataset)-(n+1):]

#distance metrics between two laternatives and a given target direction (positive or negative)
def distance(alternative,template,positive_target):
    if positive_target:
        dist = alternative-template
        dist[dist<0]=0
    else:
        dist =template-alternative
        dist[dist<0]=0
    return sum(dist)

#same as distance() applied over an array of alternatives
def distance_arr(alternative,template,positive_target):
    if positive_target:
        dist = alternative-template
        dist[dist<0]=0
    else:
        dist =template-alternative
        dist[dist<0]=0
    return np.sum(dist,axis=1)

#returns positions of attributes with differnet values than the template_numeric
def find_changes(alternative,template_numeric,positive_target):
    if positive_target:
        return np.argwhere(alternative>template_numeric)
    return np.argwhere(alternative<template_numeric)

#updates columns at index indx with the given values val from a given matrix A, updates eac
#can be optimized. e.g., to avoid geenrating duplicates
def update_per_row(A, indx, val,num_elem=1):
    all_indx = indx[:,None] + np.arange(num_elem)
    A[np.arange(all_indx.shape[0])[:,None], all_indx] =val
    return A