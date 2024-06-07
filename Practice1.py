import tensorflow as tf
import numpy as np

t=tf.zeros([5,5,5,5])
t=tf.reshape(t,[625])
t=tf.reshape(t,[125,-1])

print(t)

#ml algorithms of linear regression,classification,clustering and hidden markov rule