import math

#euclidean distance calculator
def euclidean_distance(point1, point2):
    #calculates distance between two points
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions.")

    sum_sq_diff = 0
    for i in range(len(point1)):
        sum_sq_diff += (point1[i] - point2[i])**2  #square diff

    return math.sqrt(sum_sq_diff)  #return the square root of the sum
