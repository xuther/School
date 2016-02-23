
public class MLPNode {
	
	private double[] incomingWeights; 
	private double[] lastWeights;
	private double activation;
	private double output;
	private double derivative;
	private double[] lastInputs;
	private double[] lastWeightChanges;
	private double[] bestWeights;
	
	
	private double MOMENTUM = .5;
	
	public double[] getIncomingWeights() {
		return incomingWeights;
	}

	public void setIncomingWeights(double[] incomingWeights) {
		this.incomingWeights = incomingWeights;
	}
	
	public void setBestWeights()
	{
		this.bestWeights = incomingWeights.clone();
	}
	
	public void restoreBestWeights() {
		this.incomingWeights = bestWeights.clone();
	}
	
	public MLPNode(int connections, java.util.Random rand) {
		//we initialize the weights array. 
		incomingWeights = new double[connections];
		
		
		//we initialize the defaults to small, random values between 0 and 1
		for (int i = 0; i < connections;i++) {
			incomingWeights[i] = rand.nextDouble() - 0.5;
			//incomingWeights[i] = 1;
		}
	}
	
	public MLPNode(int connections, java.util.Random rand, double Momentum) {
		this(connections, rand);
		MOMENTUM = Momentum; 
	}
	
	/**
	 * Evaluates if the current node should fire or not. Note that incoming MUST be the
	 * same size as incomingWeights
	 * 
	 * @param incoming The array of inputs from the layer before this node.
	 */
	public double Evaluate(double[] incoming) { 
		activation = 0;
		
		assert (incoming.length == incomingWeights.length);
		
		//Save the inputs for our weight update calculation. 
		lastInputs = incoming.clone();
		
		//do our dot product 
		for (int i = 0; i < incomingWeights.length; i++) {
			activation += incomingWeights[i] * incoming[i];
		}
		
		//get our activation value 
		//TODO: this is different from what was in the slides, but matches the book.
		output = 1.0/(1+Math.exp(-activation));
		
		return output;
	}
	
	//Calculate the derivative of error slope. 
	public double calculateGreekSymbolThingOutput(double target) {
		double toReturn = (target - output)* output*(1-output);
		derivative = toReturn;
		return toReturn;
	}
	
	//Calculate the derivative of error slope. 
	public double calculateGreekSymbolThing(double[] outgoingWeights, double[] derivatives) {
		assert (outgoingWeights.length == derivatives.length);
		
		double toReturn = output*(1-output);
		double sum = 0;
		
		for (int i = 0; i < derivatives.length; i++)
		{
			sum += outgoingWeights[i] * derivatives[i];
		}
		toReturn = toReturn*sum;
		
		this.derivative = toReturn;
		return toReturn;
	}
	
	//Calculate the weight changes according to Learning Rate * derivativeOfErrorSlope * LastInputs
	public double[] calculateWeightChanges(double learningRate, double greekThing) {
		double[] weightChanges = new double[incomingWeights.length];
		
		for (int i = 0; i < incomingWeights.length; i ++){
			weightChanges[i] = learningRate * greekThing * lastInputs[i];
		}
		
		lastWeightChanges = weightChanges.clone();
		return weightChanges;
	}

	//Update the weights according to oldWeight + Change + Momentum calculation. 
	public void changeWeights(double[] weightChanges) {
		assert(weightChanges.length == incomingWeights.length);
		lastWeights = incomingWeights.clone();
		
		for (int i = 0; i < incomingWeights.length; i++) {
			incomingWeights[i] = incomingWeights[i] + weightChanges[i] + (MOMENTUM * lastWeightChanges[i]) ;
		}
	}
	
	public double getDerivative(){
		return derivative;
	}
	
	public double getWeight(int incomingNode){
		return incomingWeights[incomingNode];
	}

	public double getLastWeight(int weightNo) {
		return lastWeights[weightNo];
	}

	public double getMSE(double target) {
		return Math.pow((output - target), 2);
	}
	
}
