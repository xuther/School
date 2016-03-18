import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


public class DTNode {
	Matrix values;
	Matrix labels;
	
	//These are the features AVAILABLE for split. 
	ArrayList<Integer> featuresForSplit;
	
	//The Value of the feature that was split on. 
	Integer featureSplitOn;
	
	//Maps the feature split on value to the child.  
	HashMap<Integer, DTNode> children;
	
	//For leaf nodes, the label given to the node. 
	Integer nodeLabel;
	
	DTNode parent;
	
	int id;
	
	Integer splitValue;
	
	public DTNode(Matrix values, Matrix labels, ArrayList<Integer> featuresForSplit, DTNode parent, Integer SplitValue){
		this.values = values;
		this.labels = labels;
		this.featuresForSplit = (ArrayList<Integer>) featuresForSplit.clone();
		this.children = new HashMap<Integer, DTNode>();
		this.featureSplitOn = 0;
		this.nodeLabel = Integer.MIN_VALUE;
		this.parent = parent;
		this.splitValue = SplitValue;
	}
	
	public int predict(double[] features){
		if (this.nodeLabel != Integer.MIN_VALUE){
			return this.nodeLabel;
		}
		else if (this.children.size() == 0)
			return findMostCommonValue(this.featureSplitOn);
		else{
			int col = featureSplitOn;
			DTNode child = children.get((int) Math.floor(features[col]));
			
			//Unknown value. 
			if (child == null) {
				child = children.get(findMostCommonValue(this.featureSplitOn));
			}
			if (child == null) {
				return findMostCommonValue(this.featureSplitOn);
			}
			
			return child.predict(features);
		}
	}
	
	//Recursively add all the non leaf nodes. 
	public void addNodes(ArrayList<DTNode> list) {
		if (this.nodeLabel != Integer.MIN_VALUE)
			return;
		
		
		for(DTNode d : children.values()){
			if (d.nodeLabel == Integer.MIN_VALUE){
				list.add(d);
				d.addNodes(list);
			}
		}
	}
	
	
	public int labelNodes(int id){
		this.id = id;
		int tempid = id;
		tempid ++;
		if (this.children.size() == 0)
			return tempid++;
		
		for(DTNode d : children.values()){
			tempid = d.labelNodes(tempid);
		} 
		return tempid;
	}
	
	public int calculateDepth(){
		return this.countParent();
	}
	
	public int countParent() {
		if (this.parent == null)
			return 0;
		else
			return this.parent.countParent() + 1;
	}
	public int findMostCommonValue(int featureForSplit) {
		HashMap<Integer, ArrayList<Integer>>splitLines = countValues(featureForSplit, buildRows());
		
		Iterator<Integer> it = splitLines.keySet().iterator();
		
		int curWinner = -1;
		int winningNumber = Integer.MIN_VALUE;
		while(it.hasNext()) {
			int curValue = it.next();
			int challenger = splitLines.get(curValue).size();
			
			if (challenger > winningNumber)
			{
				curWinner = curValue;
				winningNumber = challenger;
			}
		}
		return curWinner;
	}
	
	/*
	 * This is the function to build the tree. Basically evaluate each of the 
	 * potential splits - create the proper sub-nodes and then  
	 */
	public void split(){
		//if the node is all the way down, or is pure, label it. 
		if (featuresForSplit.size() == 0)
		{
			this.nodeLabel = findMostCommonLabel();
		}
		
		if(featuresForSplit.size() == 0 || findLabels().size() == 1) {
			 this.nodeLabel = findMostCommonLabel();
			return;
		}
		
		HashMap<Integer, ArrayList<Integer>> labelCount = findLabels();
		HashMap<Integer, Double> infoValues = new HashMap<Integer, Double>();
		
		Integer currentBestSplit = -1;
		Double currentBestInfo = Double.POSITIVE_INFINITY;
		
		for (Integer i : featuresForSplit) {
			infoValues.put(i, calculateInfo(i, labelCount));
			if (infoValues.get(i) < currentBestInfo) {
				currentBestSplit = i;
				currentBestInfo = infoValues.get(i);
			}
		}
		
		this.featureSplitOn = currentBestSplit;
		ArrayList<Integer> featuresForChildren = this.featuresForSplit;
		featuresForChildren.remove(currentBestSplit);
		
		HashMap<Integer, ArrayList<Integer>>splitLines = countValues(currentBestSplit, buildRows());
		
		Iterator<Integer> it = splitLines.keySet().iterator();

		//Split the nodes - give each child it's relevant matrix,
		while(it.hasNext()){
			Integer curFeature = it.next();
			ArrayList<Integer> rows = splitLines.get(curFeature);
			
			if (rows.size() == 0){
				continue;
			}


			Matrix valuesForChild = new Matrix(this.values, rows.get(0), 0, 1, this.values.cols());
			Matrix labelsForChild = new Matrix(this.labels, rows.get(0), 0, 1, this.labels.cols());
			for (int i = 1 ; i < rows.size() ; i++)
			{
				try {
					valuesForChild.add(this.values, rows.get(i), 0, 1);
					labelsForChild.add(this.labels, rows.get(i), 0, 1);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			this.children.put(curFeature, new DTNode(valuesForChild,labelsForChild,featuresForChildren, this, curFeature));
		}
		
		Iterator<Integer> itChild = children.keySet().iterator();
		
		//Reursivly call split on each child. 
		while(itChild.hasNext()){
			children.get(itChild.next()).split();
		}
		 
	}
	
	public int findDepth(int curbest){
		int thisdepth = calculateDepth();
		
		if (thisdepth > curbest)
			curbest = thisdepth;
		
		if (children.size() == 0)
			return curbest;
		else {
			for (DTNode d : children.values()){
				curbest = d.findDepth(curbest);
			}
			return curbest;
		}
		
		
	}
	
	public int findMostCommonLabel() {
		HashMap<Integer, ArrayList<Integer>> labels = findLabels();
		Iterator<Integer> it = labels.keySet().iterator();
		
		int curBest = Integer.MIN_VALUE;
		int curLabel = -1;
		while(it.hasNext()){
			int label = it.next();
			int challenger = labels.get(label).size();
			
			if (challenger > curBest){
				curBest = challenger;
				curLabel = label;
			}
		}
		
		return curLabel;
	}
	
	private HashMap<Integer, ArrayList<Integer>> findLabels() {
		return findLabels(buildRows());
	}
	
	private HashMap<Integer, ArrayList<Integer>> findLabels(ArrayList<Integer> candidates) {
		//This is a map from the label to a list of all the indicies that had that value(So we can inspect them
		HashMap<Integer, ArrayList<Integer>> v = new HashMap<Integer, ArrayList<Integer>>();  
		
		for(Integer i : candidates){
			int value = (int) Math.floor(labels.row(i)[0]);
			
			if (!v.containsKey(value)) {
				ArrayList<Integer> toAdd = new ArrayList<Integer>();
				toAdd.add(i);
				v.put(value, toAdd);
			}
			else{
				v.get(value).add(i);
			}
		}
		
		return v;
	}
	
	/*
	 * Returns a list of Feature values mapped to the row number of that value in the matrices. Assumes 
	 * values of an integer value. 
	 */
	private HashMap<Integer, ArrayList<Integer>> countValues(int ValueToCount,ArrayList<Integer> candidates) {
		HashMap<Integer, ArrayList<Integer>> v = new HashMap<Integer, ArrayList<Integer>>();
		
		for(Integer q : candidates){
			int value = (int) Math.floor(values.row(q)[ValueToCount]);
			
			if(!v.containsKey(value)){
				v.put(value, new ArrayList<Integer>());
			}
			v.get(value).add(q);
		}
		
		return v;
	}
	
	private ArrayList<Integer> buildRows(){
		ArrayList<Integer> toReturn = new ArrayList<Integer>();
		
		for(int i = 0; i < values.rows(); i++)
		{
			toReturn.add(i);
		}
		
		return toReturn;
	}
	
	private double calculateInfo(int featureToCalculate, HashMap<Integer, ArrayList<Integer>> labelCounts) {		
		//we need to build our formula - piece by piece. 
		
		HashMap<Integer, ArrayList<Integer>> featureCount = countValues(featureToCalculate,buildRows());
		Iterator<Integer> it = featureCount.keySet().iterator();
		
		double sum = 0.0;
		
		//calculate each of our 'outside' values
		while(it.hasNext())
		{
			Integer label = it.next();
			ArrayList<Integer> candidates = featureCount.get(label);
			
			//the coefficient that precedes each group. 
			double coefficient = (double)candidates.size() / values.m_data.size(); 
			
			double innerSum = 0.0;
			
			HashMap<Integer, ArrayList<Integer>> labelCount = findLabels(candidates);
			//Calculate the inner part.
			Iterator<Integer> it2 = labelCount.keySet().iterator();
			
			while(it2.hasNext())
			{
				Integer feature = it2.next();
				int count = labelCount.get(feature).size();
				
				double fraction = (double)count / candidates.size();
				double featureValue = -1 * fraction * (Math.log(fraction)/Math.log(2));
				innerSum += featureValue;
			}
			
			sum += (coefficient * innerSum);
		}
		
		return sum;
	}

	public void addAllNodes(ArrayList<DTNode> list) {
		if (this.children.size() == 0)
			return;
		
		for(DTNode d : children.values()){
				list.add(d);
				d.addAllNodes(list);
		}
		
	}
}
