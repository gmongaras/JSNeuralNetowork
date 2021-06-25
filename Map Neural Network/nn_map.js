// Puts a t value through the sigmoid function and returns the value
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}



// Returns the output when x is sent through the ReLU function
function ReLU(x) {
	if (x >= 0) {
		return x;
	}
	else {
		return 0;
	}
}




// Returns the output when x is sent through the tanh function.
function tanh(x) {
	(Math.pow(Math.E, x) - Math.pow(Math.E, -x))/(Math.pow(Math.E, x) + Math.pow(Math.E, -x))
}




// Calculates the softax given an array of values
// Link to softmax function: https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
function softmax(inputs) {
	// Calculate the top half of the funciton.
	expVals = []; // The exponential values for each item
	
	for (let i = 0; i < inputs.length; i++) {
		expVals.push(Math.pow(Math.E, inputs[i]));
	}
	
	
	// Calculate the bottom half of the function
	summation = 0; // The summation of all exponential values
	
	for (let i = 0; i < inputs.length; i++) {
		summation += expVals[i];
	}
	
	
	// Calculate the final value for each input. Each input is it's exponential
	// value divided by the summation
	outputs = []; // The final value of each input
	
	for (let i = 0; i < inputs.length; i++) {
		outputs.push(expVals[i]/summation);
	}
	
	return outputs;
}






// A layer for a neural network with the functions needed to make it operate properly. Maps
// are used to make this happen
class Layer {
	constructor(inputs, nodes, activation) {
		// Initialize the number of inputs and nodes
		this.n_inputs = inputs;
		this.n_nodes = nodes;
		
		// Initialize the activation function
		this.activation = activation;
		
		// Fill the weights array with random values
		this.weights = [];
		// For every input
		for (let i = 0; i < nodes; i++) {
			// Add a new random weight for each input connected to the node. Multiply each
			// random value by 0.01 so it's small enough for gradient descent to be efficient
			// if gradient descent is used.
			this.weights.push([...Array(inputs)].map(() => parseFloat(((Math.random() * 1) * (Math.round(Math.random()) ? 1 : -1)) * 0.01)));
		}
		
		// Initialize all biases to 0
		this.biases = [...Array(nodes)].map(() => 0);
	}
	
	
	// Given this.n_inputs number of inputs, return the values from a forward pass.
	forward(inputs) {
		// The inputs must be the same size as n_inputs
		if (this.n_inputs == inputs.length) {
			// The forward pass is found from the dot product between the weights and biases
			// added to the bias for each node.
			if (this.weights[0].length == undefined) {
				this.weights = [this.weights]
			}
			let z = ssum(mmultiply([inputs], transpose(this.weights))[0].map(Number), this.biases).map(Number);
			// Send each result through an activation function.
			if (this.activation == "sigmoid") {
				return z.map((x, i) => sigmoid(z[i]));
			}
			else if (this.activation == "ReLU") {
				return z.map((x, i) => ReLU(z[i]));
			}
			else if (this.activation == "sigmoid") {
				return z.map((x, i) => tanh(z[i]));
			}
			else if (this.activation == "softmax") {
				return softmax(z);
			}
			
			// If the activation function is not a given activation function, return the output (z).
			return z;
		}
		
		// If the inputs are not the same size, return null
		return null;
	}
}



// A class to create a neural network structure using mapping.
class NeuralNetwork {
	constructor(layers, layer_sizes, inputs) {
		// layers = number of layers. If there are 5 layers, layers = 5
		// layer_sizes = number of nodes in each layer in an array. If layers = 2,
		//				 layer_sizes = [8, 10] If the hidden layer has 8 nodes and
		//				 the output layer has 10 nodes.
		// inputs = number of inputs. If there are 6 inputs, inputs = 6
		this.l = layers;
		this.ls = layer_sizes;
		this.i = inputs;
		
		
		// Create "layers-1" number of hidden layers
		this.layers = []; // Stores each layer in the neural network
		let layerInputs = inputs
		for (let i = 0; i < layers-1; i++) {
			this.layers.push(new Layer(layerInputs, layer_sizes[i], "ReLU"));
			layerInputs = layer_sizes[i];
		}
		// Add the output layer
		this.layers.push(new Layer(layerInputs, layer_sizes[layers-1], "softmax"));
	}
	
	
	// Given an input vector of size "this.i", the function returns the output from the neural network
	predict(inputs) {
		// If the inputs are the right size
		if (inputs.length == this.i) {
			// Iterate through each layer
			let outputs = inputs;
			for (let layer of this.layers) {
				outputs = layer.forward(outputs);
			}
			
			// Return the outputs
			return outputs;
		}
		
		// Return null if the inputs are not the right size
		return null;
	}
}