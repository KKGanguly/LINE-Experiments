# Define the exponential decay function
import math
import numpy as np

def exponential_decay(value, decay_rate, time):
    return value * np.exp(-decay_rate * time)

def update_decay_rate(selected, alpha=0.1, min_decay=0.8, max_decay=0.99):
    return max(min_decay, min(max_decay, 1 - math.exp(-alpha * selected)))

# Adapted UCB function with exponential decay
def ucb_with_decay(variance, sum, sumsq , time=1, selected = 1):
    # Apply exponential decay to mean and variance
    #decayed_mean = exponential_decay(mean, decay_rate, time)
    #decayed_variance = exponential_decay(variance, decay_rate, time)
    selected = selected + 1e-10
    V = (sumsq/selected)-(sum/selected)**2 + math.sqrt((2*math.log(time)/selected))
    beta = math.sqrt((math.log(time)/selected)*min(0.25, V))
    #print("UCB Distribution=====================")
    #print(sum/selected)
    #print(beta * math.sqrt(variance))
    # Calculate UCB
    return sum/selected + beta * math.sqrt(variance)

def ucb(variance, sum, sumsq , time=1, selected = 1, decay_rate = 0.99):
    # Apply exponential decay to mean and variance
    #decayed_mean = exponential_decay(mean, decay_rate, time)
    #decayed_variance = exponential_decay(variance, decay_rate, time)
    selected = selected + 1e-10
    V = (sumsq/selected)-(sum/selected)**2 + math.sqrt((2*math.log(time)/selected))
    decay_rate = update_decay_rate(selected=selected)
    V = decay_rate * V + (1 - decay_rate) * variance 
    beta = math.sqrt((math.log(time)/selected)*min(0.25, V))
    #print("UCB Distribution=====================")
    #print(sum/selected)
    #print(beta * math.sqrt(variance))
    # Calculate UCB
    if selected < 5:  # If very few labels, downweight exploration
        beta *= 0.5  
        
    # Reduce confidence bound if variance_x is small (dense region)
    if variance < 1e-3:  
        beta *= 0.1 
    return sum/selected + beta * math.sqrt(variance)
    #return sum + math.sqrt((2*math.log(time)/selected)) * math.sqrt(variance)