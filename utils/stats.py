import numpy as np
import scipy.stats as stats

def get_mean(input):
    return round(np.mean(input), 4)

def get_std(input):
    return round(np.std(input), 4)

def get_confidence(input, percentage=0.95):
    mean_prob = get_mean(input)
    std_dev = get_std(input)
    conf_int = stats.norm.interval(percentage, loc=mean_prob, scale=std_dev/np.sqrt(len(input)))[1] - mean_prob
    return round(conf_int, 4)

def get_entropy(input):
    input = np.clip(input, 1e-10, 1 - 1e-10)
    entropy = -np.sum(input * np.log2(input) + (1 - input) * np.log2(1 - input))
    return round(entropy / len(input), 4)

def get_stats(type, input):

    if type=='mean':
        return get_mean(input)
    elif type=='std':
        return get_std(input)
    elif type=='entropy':
        return get_entropy(input)
    elif type[:5]=='conf_':
        conf_perc = float(type[5:])/100
        return get_confidence(input, percentage=conf_perc)