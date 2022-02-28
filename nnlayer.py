import matplotlib.pyplot as plt 
import cv2
import numpy as np 

choices = np.zeros([1, 36])
choices[0,5] = 1
options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',"0","1","2","3","4","5","6","7","8","9"]

#the layer dense class is the class responsible for the neurons and the calculations behind them.
class Layer_Dense:
    #init ititialises the weights and biases
    def __init__(self, n_inputs, n_neurons):#
        #the weights are generated randomly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        #forward stores the inputs for the backpropogation and then does a matrix product of the inputs and weights and then sums that with the biases
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #backward takes the derivatives fo the weights biases and inputs to calculate the gradients which will be used later when optimizing. 
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    #the rectified linear activation function takes all values that are less than 0 and sets them to 0, and any values greater than 0 stay the same on a forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        #a backwards pass makes a copy of the values
        self.dinputs = dvalues.copy()
        #the gradient of all the values less than 0 is 0, which is what is shown here
        self.dinputs[self.inputs <= 0] = 0

class Activation_SoftMax:
    def forward(self, inputs):
        #remembers the values for the backpropogation
        #calculates the exponentiated propobilites
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        #normalises them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
    def backward(self, dvalues):
        #creates an array
        self.dinputs = np.empty_like(dvalues)
        #enumerates the gradients and outputs, flattens them
        for index, (single_output, single_dvalues) in enumerate (zip (self.output, dvalues)):
            single_output = single_output.reshape( - 1 , 1 )
            #creates a jackbean matris of the outputs
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #calculates a sample wise gradient and adds it to the array of the sampe gradients. 
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

            #standard loss class
class Loss: 
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    #cross entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len (y_true.shape) == 1:
            #calculates the propabilites for the target values if labels are catagorical
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            #vMask values
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
            #losses ouptutted
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        #sample number
        samples = len(dvalues)
        #counting of the number of labels in each sample
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            #calculation of gradient and normalisation. 
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs /samples

#this class is designed to make backpropogation quicker
class Activation_Combined():
    def __init__(self):
        #creates activaiton and loss funciton objects
        self.activaiton = Activation_SoftMax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        #output layers activation fucnction
        self.activaiton.forward(inputs)
        #sets value of output
        self.output = self.activaiton.output
        #returns loss balue
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #turns one hot encoded labels into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis =1)
        self.dinputs = dvalues.copy()
        #caculates and normalises gradient
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

        #using the adaptive momentum optimizer
class Optimizer:
    #sets the settings for the optimiser and saves them
    def __init__(self, learning_rate = 0.01, decay= 0., epsilon=1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta_1
        self.beta2 = beta_2
        #called before the parameters are updated
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
            if not hasattr(layer, 'weight_cache'):
                #if the layer doesn't contain cache arrys, it fills it with zeroes.
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)
            #updates momentums with the current gradients calculated
            layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
            layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
             #performs calculations to get the corrected momentum. as the iterations are at 0 we add 1 to it.
            weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))
            #gets the corrected cache values
            layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
            layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
            weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
            #parameter update and normalisation
            layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1

letters = [4, 29, 6, 11, 4]

#initiallses the input layer
dense1 = Layer_Dense(15,64)
#initialises the first activation funciton
activation1 = Activation_ReLU()

#initialises the second layer
dense2 = Layer_Dense(64,128)
activation2 = Activation_ReLU()
#initialises the second activation funciion
dense3 = Layer_Dense(128,256)
activation3 = Activation_ReLU()
dense4 = Layer_Dense(256,36)
loss_activation = Activation_Combined()

#initlaises the optimizer, setting leanring and decay rate
optimizer = Optimizer(learning_rate = 0.055, decay=5e-7)

#trains the neural network
for iteration in range(100):

    for b in range(0, 4):
        choices = np.zeros([1, 36])
        choices[0,letters[b]] = 1
        newpath = r"D:\Downloads\archive\SegmentsFromPlate\Seg" + str(b) +'.png'
        x = cv2.imread(newpath, 0)
        y = choices

        #puts the points through the network to calculate which spiral each point is in
        dense1.forward(x)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        dense4.forward(activation3.output)

        #calculates the loss taking into account the output from the hidden layer, and the actual value
        loss = loss_activation.forward(dense4.output, y)

        #predictions for each value x 
        predictions = np.argmax(loss_activation.output, axis=1)
        if len (y.shape) == 2:
            y = np.argmax(y, axis= 1)
        #accuracy of each prediction - the amount of the guesses that were correct
        accuracy = np.mean(predictions == y)

        if not iteration % 1:
            print(f'iteration: {iteration}, ' +
                f'letter: {b}, ' +
                f'acc: {accuracy: 3f}, ' +
                f'loss: {loss:.3f}, ' +
                f'lr: {optimizer.current_learning_rate} ' )

        #backpropogation calculating gradients 
        loss_activation.backward(loss_activation.output, y)
        dense4.backward(loss_activation.dinputs)
        activation3.backward(dense4.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        #optimisation parameter update, alteration of weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.update_params(dense4)
        optimizer.post_update_params()

newpath = r"D:\Downloads\archive\SegmentsFromPlate\Seg0.png"
x = cv2.imread(newpath, 0)

dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
dense4.forward(activation3.output)
loss = loss_activation.forward(dense4.output, y)
predictions = np.argmax(loss_activation.output, axis=1)
guess = predictions[4]
print(options[guess])
