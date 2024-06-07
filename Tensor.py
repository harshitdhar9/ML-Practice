import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#creating tensors
string=tf.Variable("Hello",tf.string)
integer=yf.Variable(123,tf.int16)
float=tf.Variable(2.34,tf.float64)

#rank of tensors
rank1_tensor=tf.Variable(["Test1"],tf.string)
rank2_tensor=tf.Variable([["Test1","Ok"],["Test2","Yes"]],tf.string)

#to check rank of tensor
tf.rank(rank2_tensor)

#shape of tensors
rank2_tensor.shape

#changing shape of tensor
tensor1=tf.ones([1,2,3])
tensor2=tf.reshape(tensor1,[2,3,1])
tensor3=tf.reshape(tensor2,[3,-1])

#types: variable,constant,placeholder,sparsetensor

#evaluating tensors
with tf.Session() as sess:
    tensor.eval()  #name of tensor

