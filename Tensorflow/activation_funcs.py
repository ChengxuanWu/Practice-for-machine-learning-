#Try to realize the all activation-functions
import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import expit > to replace the logistic function
#compare the softmax and sigmoid function for classification
#practive about sigmoid or logistic function
#negtive log-likelihood function
X = np.array([1, 1.4, 2.5])
w = np.array([0.4, 0.3, 0.5])

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1/(1+np.exp(-z))
'''Output range would between 0 to 1, sotimes, the learning rate would be 
too slow to let the function stuck in a regional minimum'''
def logistic_activation(X,w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = %.3f' % logistic_activation(X, w))

#multiple activated function
#this weighted array have the shape of out_units and n_hidden_units+plus 1 bias
#So 3 x 4 is the shape if this array
w = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])
A = np.array([[1, 0.1, 0.4, 0.6]])
z = np.dot(w, A[0])
y_probas = logistic(z)
print('Net Inputs: \n', z)
print('Output Units:\n', y_probas)
#here shows the probabilities: but the sum isn't equal to 1
#still can use np.argmax to do prediction for classification
print('predicted class label:',np.argmax(y_probas, axis=0))

#try to use softmax to get probabilities of respective class
#the sum of all probabilities is 1.0
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

y_probas = softmax(z)
print('Probabilities:\n', y_probas)

#Another sigmoid function: hyperbolic tangent (tanh)
'''widen the output range to -1 and 1, being easier to deal with the reginal
minimum problem'''
#function: tanh(z) = 2*logistic(z)-1
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0., color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
plt.plot(z, log_act, linewidth=3, label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
#the function logistic could be replaced by scipy.special.expit
#tanh could be replaced by np.tanh
