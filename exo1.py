import numpy as np

def sigmoid(x) :
	return 1 / (1 + np.exp(-x))

def dsigmoid(y) :
	return y * (1.0 * y)

class NeuralNetwork(object):
	def __init__(self, unit) :
		"""
		:param unit : [5, 9, 7, 2] means 4 layers => 5 units in input layer, 2 hidden layers (9 units and 7 units), and 2 units in output layer
		"""
		# Number of units in each layers
		self.input = unit[0] + 1	# 1 for the bias
		self.output = unit[len(unit) - 1]
		self.hidden = np.zeros(len(unit) - 2)
		for i in range(len(unit) - 1):
			self.hidden[i] = unit[i+1] + 1	# 1 for the bias

		# number of hidden layers
		self.nbHiddenLayer = len(self.hidden)
		
		# Init a (value in unit)
		self.aInput = [1.0] * self.input
		self.aOutput = [1.0] * self.output
		self.aHidden = [1.0] * self.hidden
		
		# Init the weighs randomly in matrix
		self.wInputToHidden = np.random.randn(self.input, self.hidden[0])
		self.wHiddenToOutput = np.random.randn(self.hidden[self.nbHiddenLayer] - 1, self.output)
		self.wHiddenToHidden = np.zeros(self.nbHiddenLayer - 1)
		for i in range(self.nbHiddenLayer - 1) :
			self.wHiddenToHidden[i] = np.random.randn(self.hidden[i], self.hidden[i+1])


	# feefForward begin #
	def feedForward(self, inputs):
		if len(inputs) != self.input - 1:
			raise ValueError('Wrong dimension for inputs')

		# Calculate the units inputs
		inputs.append(1)
		self.aInput = inputs

		# Calculate the units of the first hidden layers
		self.aHidden[0] = sigmoid(np.dot(self.wInputToHidden, self.aInput))

		# Calculate the units of the others hidden layers
		for i in range(self.nbHiddenLayer - 2):
			self.aHidden[i+1] = sigmoid(np.dot(self.wHiddenToHidden[i], self.aHidden[i]))

		# Calculate the units output
		self.aOutput = sigmoid(np.dot(self.wHiddenToOutput, self.aHidden[self.nbHiddenLayer - 1]));

		return self.aOutput
	# feedForward end #


	# backPropagation begin #
	def backPropagation(self, target, alpha) :
		if len(outputs) != self.output :
			raise ValueError('Wrong dimension for outputs')

		# delta for output
		delta_outputs = [0.0] * self.output
		# calculate error for output
		error = selt.output - target
		delta_outputs = error * dsigmoid(self.aOutput)

		# delta for hidden (all layers)
		delta_hiddens = [0.0] * self.hidden
		# calcultate error for the last hidden layer
		error = np.dot(self.wHiddenToOutput, delta_outputs)
		delta_hiddens[self.nbHiddenLayer - 1] = error * dsigmoid(self.aHidden[self.nbHiddenLayer - 1])
		# calculate error for the other hidden layers
		for i in range(self.nbHiddenLayer - 2).reverse() :
			error = nb.dot(self.wHiddenToHidden[i], delta_hiddens[i+1])
			delta_hiddens[i] = error * dsigmoid(self.aHidden[i])

		# delta for input
		delta_inputs = [0.0] * self.input
		# calculate error for the input layer
		error = np.dot(self.wInputToHidden, self.aInput)
		delta_inputs = error * dsigmoid(self.aInput)
		# BEGIN TODO #
		# END TODO #
	# backPropagation end #




















