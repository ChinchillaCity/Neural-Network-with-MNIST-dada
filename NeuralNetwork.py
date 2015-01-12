import random
import load
import sys
import numpy as np

class NeuralNetwork():

    def __init__(self):
        sizes=[784, 30, 10]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        training_data, test_data=load.start(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[3])
        self.useSGD(training_data, 30, 10, 0.3, test_data,sys.argv[4])

    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def useSGD(self, training_data, epochs, mini_batch_size, eta,
            test_data,outputfile):
        
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMinibatch(mini_batch, eta)
        self.test(test_data,outputfile)

    def updateMinibatch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def test(self, test_data, outputfile):
        test_results = [(np.argmax(self.feedForward(x)), y) 
                        for (x, y) in test_data]
        
        f = open(outputfile, 'w')
        for (x, y) in test_results:
           f.write(str(x)+"\n")
        f.close
      
    def cost_derivative(self, output_activations, y):
        return (output_activations-y) 

def sigmoid(z):
    #Sigmoid function
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    #Derivative of sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def main():
        NeuralNetwork()
        
if __name__ == '__main__':
        main()
