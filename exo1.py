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
		self.input = unit[0]
		self.output = unit[len(unit) - 1]
		for i in range(len(unit) - 1) :
			self.hidden[i] = unit[i+1]

		# number of hidden layers
		self.nbHiddenLayer = len(self.hidden)
		
		# Init a (value in unit)
		self.aInput = [1.0] * self.input
		self.aOutput = [1.0] * self.output
		for i in range(len(self.hidden) - 1) :
			self.aHidden[i] = [1.0] * self.hidden
		
		# Init the weighs randomly
		self.wInputToHidden = np.random.randn(self.input, self.hidden[0])
		self.wHiddenToOutput = np.random.randn(self.hidden[self.nbHiddenLayer] - 1, self.output)
		for i in range(len(self.hidden) + 1) :
			self.wHiddenToHidden = np.random.randn(self.hidden[i], self.hidden[i+1])

	# feefForward begin #
	def feedForward(self, inputs):
		if len(inputs) != self.input :
			raise ValueError('Wrong dimension for inputs')

		# Calculate the units inputs
		for i in range(self.input) :
			self.aInput[i] = inputs[i]

		# Calculate the units of the first hidden layers
		for  i in range(self.hidden[0]) :
			sum = 0.0
			for j in range(self.input) :
				sum += self.aInput[i] * self.wInputToHidden[i][j]
			self.aHidden[0][i] = sigmoid(sum)

		# Calculate the units of the others hidden layers
		for i in range(self.nbHiddenLayer - 1) :	# For each hidden layer
			for j in range(self.hidden[i + 1]) :
				sum = 0.0
				for k in range(self.hidden[i]) :
					sum += self.aHidden[j][k] * self.wHiddenToHidden[j][i][k]
				self.aHidden[j] = sigmoid(sum)

		# Calculate the units output
		for j in range(self.output) :
			sum = 0.0
			for k in range(self.hidden[self.nbHiddenLayer - 1]):
				sum += self.aHidden[self.nbHiddenLayer - 1][k] * self.wHiddenToOutput[j][k]
			self.aOutput[j] = sigmoid(sum)

		return self.aOutput
	# feedForward end #

	# backPropagation begin #
	def backPropagation(self, outputs, alpha) :
		if len(outputs) != self.output :
			raise ValueError('Wrong dimension for outputs')

		# calculate error for output
		delta_outputs = [0.0] * self.output
		for i in range(self.output) :
			error = self.output - outputs[i]
			delta_outputs[i] = error * dsigmoid(self.aOutput[i])

		# delta for hidden (all layers)
		delta_hiddens = [0.0] * self.hidden

		# calcultate error for the last hidden layer
		for i in range(self.hidden[self.nbHiddenLayer - 1]) :
			error = 0.0
			for j in range(self.output) :
				error += delta_outputs[j] * self.wHiddenToOutput[i][j]
			delta_hiddens[self.nbHiddenLayer - 1][i] = error * dsigmoid(self.aHidden[self.nbHiddenLayer - 1])

		# calculate error for the other hidden layers
		for i in range(self.nbHiddenLayer - 2).reverse() :		# last to first
			for j in range(self.hidden[i]) :
				error = 0.0
				for k in range(self.hidden[i + 1]) :
					error += delta_hiddens[i + 1] * self.wHiddenToHidden[i+1][j][k]
				delta_hiddens[i][j] = error * dsigmoid(self.aHidden[i])

		# delta for input
		delta_inputs = [0.0] * self.input

		# calculate error for the input layer
		for i in range(self.input) :
			error = 0.0
			for j in range(self.output[0]) :
				error += delta_hiddens[0][j] * self.wInputToHidden[i][j]
			delta_inputs[i] = error * dsigmoid(self.aInput[i])

		# BEGIN TODO #
		# END TODO #
	# backPropagation end #




















