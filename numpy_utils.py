import numpy as np

def generate_random_alternatives(dataset, n):
    # return dataset[random.choices(range(len(dataset)), k=n)
    #getting the last n as "random"
    return dataset[len(dataset)-(n+1):]

#distance metrics between two laternatives and a given target direction (positive or negative)
def distance(alternative,template):
    distance = np.abs(alternative-template)
    return sum(distance)

#same as distance() applied over an array of alternatives
def distance_arr(alternatives,template):
    distance = np.abs(alternatives-template)
    return np.sum(distance, axis=1)

#returns positions of attributes with differnet values than the template_numeric
#TODO: return indexes where we can improve distance, and their direction
def find_changes(alternative,initial_instance):
    indixes_positive = alternative<initial_instance
    indixes_negative = alternative>initial_instance
    return indixes_positive, indixes_negative

#updates columns at index indx with the given values val from a given matrix A, updates eac
#can be optimized. e.g., to avoid geenrating duplicates
def update_per_row(A, indx, val,num_elem=1):
    all_indx = indx[:,None] + np.arange(num_elem)
    A[np.arange(all_indx.shape[0])[:,None], all_indx] =val
    return A