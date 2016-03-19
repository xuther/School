package Main.SupervisedLearners;

import java.util.ArrayList;
import Main.Matrix;

public class MLPLayer {
	
	private int layerSizeWBias;
	private int layerSize;
	private int prevLayerSizeWBias;
	private double[] outputs;
	private ArrayList<MLPNode> nodes;
	private static double BIASINIT = 1;
	
	
	public MLPLayer(int prevLayerSize, int layerSize, java.util.Random rand) {
		this.layerSize = layerSize;
		layerSizeWBias = layerSize+1;
		prevLayerSizeWBias = prevLayerSize+1;
		
		nodes = new ArrayList<MLPNode> ();
		outputs = new double[layerSizeWBias];
		
		//Set up our neurons
		for (int i = 0; i < layerSize; i ++){
			nodes.add(new MLPNode(prevLayerSizeWBias, rand));
		}
	}
	
	//Save the best weights
	public void setBest()
	{
		for (MLPNode n: nodes){
			n.setBestWeights();
		}
	}
	
	public void restoreBest()
	{
		for (MLPNode n: nodes){
			n.restoreBestWeights();
		}
	}
	
	//We assume that the bias is the first item in the array.
	public static double[] removeBias(double[] inputs) {
		assert(inputs.length >= 2);
		
		double[] toReturn = new double[inputs.length -1];
		for (int i = 1; i < inputs.length; i++) {
			toReturn[i-1] = inputs[i];
		}
		return toReturn;
	}
	
	public static double[] addBias(double[] rawInputs) {
		double[] out = new double[rawInputs.length +1];
		
		for (int i = 0; i < rawInputs.length; i++){
			out[i+1] = rawInputs[i];
		}
		out[0] = BIASINIT;
		
		return out;
	}
	
	public double[] runLayer(double[] rawInputs)
	{
		double[] Inputs;
		
		assert (rawInputs.length == prevLayerSizeWBias || rawInputs.length == prevLayerSizeWBias - 1);
		
		//Add the bias if we need to (for first layer)
		if (rawInputs.length == prevLayerSizeWBias - 1)
			Inputs = addBias(rawInputs);
		else
			Inputs = rawInputs;
		
		for (int i = 1; i < layerSizeWBias; i++){
			outputs[i] = nodes.get(i-1).Evaluate(Inputs);
		}
		
		outputs[0] = BIASINIT;
		
		return outputs;
	}
	
	public ArrayList<MLPNode> getNodes()
	{
		return nodes;
	}
	
	public MLPNode getNode(int index)
	{
		return nodes.get(index);
	}

	public double[] getDerivatives() {
		double[] derivatives = new double[nodes.size()];
		
		for (int i = 0; i < nodes.size(); i++) {
			derivatives[i] = nodes.get(i).getDerivative();
		}
		
		return derivatives;
	}
	
	public double[] getWeights(int weightNo) {
		double[] weights = new double[nodes.size()];
		
		for(int i = 0; i < nodes.size(); i ++){
			weights[i] = nodes.get(i).getWeight(weightNo);
		}
		return weights;
	}

	public double[] getLastWeights(int weightNo) {
		double[] lastWeights = new double[nodes.size()];
		
		for (int i = 0; i < nodes.size(); i++){
			lastWeights[i] = nodes.get(i).getLastWeight(weightNo);
		}
		return lastWeights;
	}
}
