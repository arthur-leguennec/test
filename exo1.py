import numpy as np

def sigmoid(x) :
	return 1 / (1 + np.exp(-x))

def dsigmoid(y) :
	return y * (1.0 * y)

class NeuralNetwork(object):
	def __init__(self, unit) :
<<<<<<< HEAD
		"""
		:param unit : [5, 9, 7, 2] means 4 layers => 5 units in input layer, 2 hidden layers (9 units and 7 units), and 2 units in output layer
		"""
		# Number of units in each layers
		self.input = unit[0]
		self.output = unit[len(unit)]
		for i in range(len(unit)) :
			self.hidden[i] = unit[i+1]
		
		# Init a (value in unit)
		self.aInput = [1.0] * self.input
		self.aOutput = [1.0] * self.output
		for i in range(len(self.hidden) - 1) :
			self.aHidden[i] = [1.0] * self.hidden
		
		# Init the weighs randomly
		self.wInputToHidden = np.random.randn(self.input, self.hidden[0])
		self.wHiddenToOutput = np.random.randn(self.hidden[len(self.hidden)], self.output)
		for i in range(len(self.hidden) + 1) :
			self.wHiddenToHidden = np.random.randn(self.hidden[i], self.hidden[i+1])

=======
		self.input = unit[0];

		for i in range(len(unit)) :
			self.hidden[i] = unit[i+1]
		
		self.output = unit[len(unit)]

		self.aInput = [1.0] * self.input

		for i in range(len(self.hidden) - 1) :
			self.aHidden[i] = [1.0] * self.hidden
		self.aOutput = [1.0] * self.output

		self.wInputToHidden = np.random.randn(self.input, self.hidden[0])

		for i in range(len(self.hidden) + 1) :
			self.wHiddenToHidden = np.random.randn(self.hidden[i], self.hidden[i+1])

		self.wHiddenToOutput = np.random.randn(self.hidden[len(self.hidden)], self.output)

>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
	# feefForward begin #
	def feedForward(self, inputs):
		if len(inputs) != self.input - 1 :
			raise ValueError('Wrong dimension for inputs')

<<<<<<< HEAD
		# Calculate the units inputs
		for i in range(self.input) :
			self.aInput[i] = inputs[i]

		# Calculate the units of the first hidden layers
=======
		# Calcul des entrees
		for i in range(self.input) :
			self.aInput[i] = inputs[i]

		# Calcul de la premiere couche cachee
>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
		for  i in range(self.hidden[0]) :
			sum = 0.0
			for j in range(self.input) :
				sum += self.aInput[i] * self.wInputToHidden[i][j]
			self.aHidden[0][i] = sigmoid(sum)

<<<<<<< HEAD
		# Calculate the units of the others hidden layers
=======
		# Calcul des autres couches cachees (sauf la derniere)
>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
		for i in range(len(self.hidden) - 1) :
			for j in range(self.hidden[i + 1] - 1) :
				sum = 0.0
				for k in range(self.hidden[i]) :
					sum += self.aHidden[j][k] * self.wHiddenToHidden[j][i][k]
				self.aHidden[j] = sigmoid(sum)

<<<<<<< HEAD
		# Calculate the units output
=======
		# Calcul de la couche de sortie
>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
		for j in range(self.output - 1) :
			sum = 0.0
			for k in range(self.hidden[len(self.aHidden) - 1]):
				sum += self.aHidden[len(self.aHidden) - 1][k] * self.wHiddenToOutput[j][k]
			self.aOutput[j] = sigmoid(sum)

		return self.aOutput
	# feedForward end #

	# backPropagation begin #
	def backPropagation(self, outputs, alpha) :
		if len(outputs) != self.output - 1 :
			raise ValueError('Wrong dimension for outputs')

<<<<<<< HEAD
		# calculate error for output
=======
>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
		delta_outputs = [0.0] * self.output
		for i in range(self.output) :
			error = self.output - outputs[i]
			delta_outputs[i] = error * dsigmoid(self.aOutput[i])

<<<<<<< HEAD
		# calculate error for hidden (all layer)
=======
>>>>>>> 586bcf1662a2621679914c1725b9fb4f1768811a
		delta_hiddens = [0.0] * self.hidden
	# backPropagation end #




















