import java.util.ArrayList;
import java.util.HashMap;


public class DecisionTree extends SupervisedLearner {
	/**
	 * Set this for pruning to happen. 
	 */
	public boolean prune = false;
	/**
	 * Percent of data used for validation set. 
	 */
	double validationPercent = .25;
	
	
	boolean print = false;
	
	Matrix validateFeatures;
	Matrix validateLabels;
	
	DTNode head;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		
		ArrayList<Integer> columns = new ArrayList<Integer>();
		
		for (int i = 0; i < features.m_enum_to_str.size(); i++)
		{
			columns.add(i);
		}
		
		//If we're pruning pull out the validation set. 
		if (prune){
			int trainSize = (int)(validationPercent * features.rows());
			Matrix trainFeatures = new Matrix(features, trainSize, 0, features.rows() - trainSize, features.cols());
			Matrix trainLabels = new Matrix(labels, trainSize, 0, labels.rows() - trainSize, 1);
			validateFeatures = new Matrix(features, 0, 0, trainSize, features.cols());
			validateLabels = new Matrix(labels, 0, 0, trainSize, 1);
			
			
			head = new DTNode(trainFeatures, trainLabels, columns, null,null);
			
			head.split();
			
			head.labelNodes(0);
			
			pruneTree(validateFeatures, validateLabels);
			
		} else
		{
			head  = new DTNode(features, labels, columns,null,null);
			head.split();
			ArrayList<DTNode> treeCount = new ArrayList<DTNode>();
			
			treeCount.add(head);
			head.addAllNodes(treeCount);
			System.out.println("Nodes=" + treeCount.size());
			System.out.println("Depth=" + head.findDepth(-1));
		}
	}
	
	public double runTest(){
		try {
			return this.measureAccuracy(validateFeatures, validateLabels, null);
		} catch (Exception e) {
			e.printStackTrace();
			return Double.NEGATIVE_INFINITY;
		}
	}
	
	private void pruneTree(Matrix testFeatures, Matrix testLabels){
		double baseAccuracy;
		double curAccuracy;
		int curNodeIndex;
		int curNodeDepth;
		ArrayList<DTNode> treeNodes = null;
		
		
		ArrayList<DTNode> treeCount = new ArrayList<DTNode>();
		
		treeCount.add(head);
		head.addAllNodes(treeCount);
		if (!print) {
			System.out.println("Nodes=" + treeCount.size());
		}
			System.out.println("Number of Nodes in tree before prune: " + treeCount.size());
			System.out.println("Depth before prune =" + head.findDepth(-1));
			
			printTree();
		try 
		{
			do 
			{
				//base accuracy. 
				baseAccuracy = runTest();
				curAccuracy = baseAccuracy;
				curNodeIndex = -1;
				curNodeDepth = Integer.MAX_VALUE;
				
				treeNodes = new ArrayList<DTNode>();
				
				treeNodes.add(head);
				head.addNodes(treeNodes);
				
				//test each possible node deletion - take the one with the best accuracy (including our base)
				for (int i = 1; i < treeNodes.size(); i++){
					DTNode curNode = treeNodes.get(i);
					
					curNode.nodeLabel = curNode.findMostCommonLabel();
					
					double challengerAccuracy = runTest();
					if (challengerAccuracy > curAccuracy || (challengerAccuracy == curAccuracy && curNodeDepth > curNode.calculateDepth())){
						curNodeIndex = i;
						curAccuracy = challengerAccuracy;
						curNodeDepth = treeNodes.get(i).calculateDepth();
					}
					//change it back to a regular node. 
					
					curNode.nodeLabel = Integer.MIN_VALUE;
				}
				
				//There were improvements over our base accuracy, delete the children, label the node, and re-run. 
				if (curNodeIndex != -1)
				{
					treeNodes.get(curNodeIndex).nodeLabel = treeNodes.get(curNodeIndex).findMostCommonLabel();
					treeNodes.get(curNodeIndex).children.clear();
				}
			} while(curNodeIndex != -1);
						
		} catch (Exception e) {
			e.printStackTrace();
		}
		treeCount.clear();
		treeCount.add(head);
		head.addAllNodes(treeCount);
			System.out.println("Number of Nodes in tree after prune: " + treeCount.size());
			System.out.println("Depth after prune =" + head.findDepth(-1));
			printTree();
	}
	
	public void printTree(){
		if (!print)
			return;
		System.out.println("Tree: ");
		ArrayList<DTNode> treeNodes = new ArrayList<DTNode>();
		String curLine = "";
		treeNodes.add(head);
		head.addAllNodes(treeNodes);
		
		for (int i = 0; i < treeNodes.size(); i++){
			DTNode curNode = treeNodes.get(i);
			
			if (curNode.children.size() == 0)
			{
				curLine = curNode.id + " --" +curNode.nodeLabel + "--> ";
				curLine += "LEAF";
			}
			else {
				curLine = curNode.id + " --" +curNode.featureSplitOn + "--> ";
				for (DTNode d: curNode.children.values())
				{
					curLine += d.splitValue + ":" + d.id + " , ";	
				}
			}
			
			System.out.println(curLine);
		}
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		int value = head.predict(features);
		for(int i = 0; i < labels.length; i++)
			labels[i] = value;
	}
	
}
