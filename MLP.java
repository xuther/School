import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;


public class MLP extends SupervisedLearner{
	
	private ArrayList<MLPLayer> layers;
	private double LEARNINGRATE = .1;
	private double VALIDATIONPERCENT = .2;
	
	public MLP(int LayersNodes[]) {
		java.util.Random rand = new java.util.Random();
		
		layers = new ArrayList<MLPLayer>();
		
		//We initialize our layers with the number of nodes per layers, as passed into via LayersNodes
		for (int i = 0; i < LayersNodes.length; i++) {
			
			int prevLayerSize;
			if (i == 0)
				prevLayerSize = LayersNodes[i];
			else 
				prevLayerSize = LayersNodes[i-1];
			
			layers.add(new MLPLayer(prevLayerSize, LayersNodes[i], rand));
		}
	}
	
	public double[] evalutate(double[] inputs) {
		//We assume there is no Bias coming into evaluate. The Bias will be added by the first
		//layer and will come out of that layer automatically, so we just set outputs to be a bit larger. 
		//We'll remove the bias at the end before sending it back. 
		
		double[] outputs = new double[inputs.length +1];
		
		for(int i = 0; i < layers.size(); i++){
			outputs = layers.get(i).runLayer(inputs);
			//switch the inputs to the outputs, and then run it again. 
			inputs = outputs;
		}
		
		//Remove the Bias from outputs. 
		double[] toReturn = MLPLayer.removeBias(outputs);
		
		return toReturn;
	}
	
	public double updateWeights(double[] outputs, double[] targets){
		assert(outputs.length == targets.length);
		double toReturn = 0;
		//start at the back layer and work our way forward. 
		for (int i = layers.size()-1; i >= 0; i--){	
			//It's an output node layer. 
			if (i == layers.size()-1) {
				for (int j = 0; j < layers.get(i).getNodes().size(); j++){
					double greekThing = layers.get(i).getNode(j).calculateGreekSymbolThingOutput(targets[j]);
					double[] weightChanges = layers.get(i).getNode(j).calculateWeightChanges(LEARNINGRATE, greekThing);
					layers.get(i).getNode(j).changeWeights(weightChanges);
					toReturn = layers.get(i).getNode(j).getMSE(targets[j]);
				}
			}
			else {
				for (int j = 0; j < layers.get(i).getNodes().size(); j++){
					MLPLayer prevLayer = layers.get(i+1);
					
					//get the derivatives of error for each node in the next layer. 
					double[] derivatives = prevLayer.getDerivatives();
					//Get the weights going from the current nodes to all nodes in the next layer (prev in backpropagation) We need to do 
					//a plus one since 0 is always the bias weight.
					double[] weights = prevLayer.getLastWeights(j+1);
					
					double derivative = layers.get(i).getNode(j).calculateGreekSymbolThing(weights, derivatives);
					double[] weightChanges = layers.get(i).getNode(j).calculateWeightChanges(LEARNINGRATE, derivative);
					layers.get(i).getNode(j).changeWeights(weightChanges);
				}
			}
		}
		return toReturn;
	}
	
	public static void main(String[] args){
		int[] nodesLayers = {2,1};
		
		//Stuff for XOR.
		double[] inputs1 = {0,0};
		double[] inputs2 = {0,1};
		double[] inputs3 = {1,0};
		double[] inputs4 = {1,1};
		
		double[] outputs1 = {0};
		double[] outputs2 = {1};
		double[] outputs3 = {1};
		double[] outputs4 = {0};
		
		
		ArrayList<double[]> ExpectedOutputs = new ArrayList<double[]>();
		ExpectedOutputs.add(outputs1);
		ExpectedOutputs.add(outputs2);
		ExpectedOutputs.add(outputs3);
		ExpectedOutputs.add(outputs4);
		
		ArrayList<double[]> Inputs = new ArrayList<double[]>();
		Inputs.add(inputs1);
		Inputs.add(inputs2);
		Inputs.add(inputs3);
		Inputs.add(inputs4);
		
		ArrayList<double[]> actualOutputs = new ArrayList<double[]>();
		
		int[] layers = {2,1};
		MLP perceptron = new MLP(layers);
		
		double error;
		for (int j = 0; j < 5000; j++) {
			error = 0;
			for (int i = 0; i < Inputs.size(); i++) {
				double[] outputs = perceptron.evalutate(Inputs.get(i));
				error += perceptron.updateWeights(outputs, ExpectedOutputs.get(i));
				actualOutputs.add(outputs);
			}
			System.out.println(error/4.0);
		}

	}
		
	public void writeToFile(ArrayList<double[]> toWrite){
		try {
			PrintWriter writer = new PrintWriter("data.txt", "UTF-8");
			for(int i = 0; i< toWrite.size(); i++)
			{
				writer.println(toWrite.get(i)[0] + ", " + toWrite.get(i)[1]); 
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		int size = features.m_data.size();
		int validationSize = (int) Math.round(size * VALIDATIONPERCENT);
		
		Matrix validationSet = new Matrix(features, size-validationSize, 0, validationSize, features.cols());
		Matrix validationLabels = new Matrix(labels, size-validationSize, 0, validationSize, labels.cols());
		
		Matrix TrainingSet = new Matrix(features, 0, 0, size-1, features.cols());
		Matrix TrainingLabels = new Matrix(labels, 0, 0, size-1, labels.cols());
		
		boolean done = false;
		int count = 0;
		int countSansUpdate = 0;
		double error;
		double BWSF = Double.POSITIVE_INFINITY;
		ArrayList<double[]> MSE = new ArrayList<double[]>();
		
		//Our window for BSSF is 50 and we run at least 500 times. 
		while (countSansUpdate < 50 && count < 500)
		{
			error = 0;
			for (int i = 0; i < TrainingSet.rows(); i++) {
				double[] outputs = evalutate(TrainingSet.m_data.get(i));
				error += updateWeights(outputs, TrainingLabels.m_data.get(i));
			}
			double MSETrain = error/TrainingSet.rows();
			double[] realoutput = new double[1];
			
			error = 0;
			for (int i = 0; i < validationSet.rows(); i++) {
				predict(validationSet.m_data.get(i), realoutput);
				double littleError = 0;
				
				for (int j = 0; j < validationLabels.cols(); j++){
					littleError+= Math.pow(realoutput[j]-validationLabels.m_data.get(i)[j], 2);
				}
				error += littleError/validationLabels.cols();
			}
			double MSEValidation = error/validationSet.rows();
			
			System.out.println(MSETrain + " , " + MSEValidation);
			double[] mse = {MSETrain, MSEValidation};
			MSE.add(mse);
			
			count++;
			if (MSEValidation < BWSF){
				BWSF = MSEValidation;
				countSansUpdate = 0;
			}
			else
				countSansUpdate++;	
			
		}
		
		writeToFile(MSE);
		System.out.println("Epochs: " + count);
		
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double[] outputs = evalutate(features);
		for(int i = 0; i < labels.length; i++)
			labels[i] = outputs[i];
	}
}


