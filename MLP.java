import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;


public class MLP extends SupervisedLearner{
	
	private ArrayList<MLPLayer> layers;
	private double LEARNINGRATE = .05;
	private int FEATURENUM = 11;
	private double VALIDATIONPERCENT = .2;
	private int MAXITERATION = 50000;
	HashMap<Double, double[]> translateLables;
	
	public MLP(int LayersNodes[]) {
		java.util.Random rand = new java.util.Random();
		
		layers = new ArrayList<MLPLayer>();
		
		//We initialize our layers with the number of nodes per layers, as passed into via LayersNodes
		for (int i = 0; i < LayersNodes.length; i++) {
			
			int prevLayerSize;
			if (i == 0)
				prevLayerSize = FEATURENUM;
			else 
				prevLayerSize = LayersNodes[i-1];
			
			layers.add(new MLPLayer(prevLayerSize, LayersNodes[i], rand));
		}
	}
	
	public void setBestWeights()
	{
		for(MLPLayer l: layers){
			l.setBest();
		}
	}
	
	public void restoreBestWeights()
	{
		for(MLPLayer l: layers){
			l.restoreBest();
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
					toReturn += layers.get(i).getNode(j).getMSE(targets[j]);
				}
				toReturn = toReturn/layers.get(i).getNodes().size();
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
		
	public void writeToFile(ArrayList<double[]> toWrite){
		try {
			PrintWriter writer = new PrintWriter("data.txt", "UTF-8");
			for(int i = 0; i< toWrite.size(); i++)
			{
				writer.println(toWrite.get(i)[0] + ", " + toWrite.get(i)[1] + ", " + toWrite.get(i)[2]); 
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
		
		translateLables = new HashMap<Double, double[]>();
		int optionCount= labels.m_str_to_enum.get(0).size();
		
		for (int i = 0; i < optionCount; i ++)
		{
			double[] value = new double[optionCount];
			value[i] = 1;
			translateLables.put((double)i, value);
		}
		
		Matrix validationSet = new Matrix(features, size-validationSize, 0, validationSize, features.cols());
		Matrix validationLabels = new Matrix(labels, size-validationSize, 0, validationSize, labels.cols());
		
		Matrix TrainingSet = new Matrix(features, 0, 0, size-1, features.cols());
		Matrix TrainingLabels = new Matrix(labels, 0, 0, size-1, labels.cols());
		
		boolean done = false;
		int count = 0;
		int countSansUpdate = 0;
		double MSEValidation = 0;
		double error;
		double BWSF = Double.POSITIVE_INFINITY;
		ArrayList<double[]> MSE = new ArrayList<double[]>();
		
		//Our window for BSSF is 50 and we run at Most MAXITERATION
		while ((countSansUpdate < 10 && count < MAXITERATION))
		{
			error = 0;
			for (int i = 0; i < TrainingSet.rows(); i++) {
				double[] outputs = evalutate(TrainingSet.m_data.get(i));
				
				//We need to translate between their labels to ours;	
				double value = TrainingLabels.m_data.get(i)[0];
				double[] target = translateLables.get(value);
				error += updateWeights(outputs, target);
			}
			double MSETrain = error/TrainingSet.rows();
			//We have that many output nodes. 
			
			error = 0;
			int VScorrect = 0;
			for (int i = 0; i < validationSet.rows(); i++) {
				double[] realoutput = predictMSE(validationSet.m_data.get(i));
				
				//This is for metrics only.
				double result = evaluateWinner(realoutput);
				double expected = validationLabels.m_data.get(i)[0];
				
				if (result == expected)
					VScorrect++;
				
				double littleError = 0;
				
				double[] ourLabel = translateLables.get(expected);
				
				for (int j = 0; j < ourLabel.length; j++){
					//we need to translate from ValidationLabels (1,2,3) to our labels (001,010,100)
					littleError+= Math.pow(realoutput[j]-ourLabel[j], 2);
				}
				error += littleError/validationLabels.cols();
			}
			MSEValidation = error/validationSet.rows();
			double accuracry = (double)VScorrect/validationSet.rows();
			
			System.out.println(count + "," + MSETrain + ", " + MSEValidation + ", " + accuracry);
			double[] mse = {MSETrain, MSEValidation, accuracry};
			MSE.add(mse);
			
			count++;
			
			//save our weights
			if (MSEValidation < BWSF){
				setBestWeights();
				BWSF = MSEValidation;
				countSansUpdate = 0;
			}
			else
				countSansUpdate++;	
			
		}
		
		//Restore the best weights
		if(countSansUpdate > 0)
			restoreBestWeights();
		System.out.println("MSE Validation Best: " + MSEValidation);
		
		writeToFile(MSE);
		System.out.println("Epochs: " + count);
		
		
	}
	
	public int evaluateWinner(double[] outputs)	{
		int curWinner = -1;
		double curBest = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < outputs.length; i++)
		{
			if (outputs[i] > curBest){
				curWinner = i; 
				curBest = outputs[i];
			}
		}
		return curWinner;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double[] outputs = evalutate(features);
		
		//round the outputs to the nearest value.
		//for our purposes (where only one can fire at a time) we take the one with the largest value and 
		//have it 'fire'
		int curWinner = -1;
		double curBest = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < outputs.length; i++)
		{
			if (outputs[i] > curBest){
				curWinner = i; 
				curBest = outputs[i];
			}
		}
		
		for(int i = 0; i < labels.length; i++)
			labels[i] = curWinner;
	}
	
	public double[] predictMSE(double[] features) throws Exception {
		double[] outputs = evalutate(features);
		return outputs;
	}
}


