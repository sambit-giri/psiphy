import numpy as np
from scipy.spatial.distance import *

def kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div[np.isfinite(kl_div)])

def js_distance(p_probs, q_probs):
	kl1 = kl_divergence(p_probs, q_probs)
	kl2 = kl_divergence(q_probs, p_probs)
	return (kl1+kl2)/2.