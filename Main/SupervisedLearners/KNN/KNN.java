package Main.SupervisedLearners.KNN;

import Main.Matrix;
import Main.SupervisedLearners.SupervisedLearner;

import java.util.PriorityQueue;
import java.util.HashMap;
/**
 * Created by joseph on 3/18/16.
 */
public class KNN extends SupervisedLearner {

    private Matrix datafeatures;
    private Matrix datalabels;
    private boolean _weighting;
    private int _k;

    public KNN(boolean weighting, int k)
    {
        _weighting = weighting;



        //we can't have a value of k less than 1.
        if (k < 1)
            _k = 1;
        else
            _k = k;
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        //Nothing to do here, since we just worry about finding the nearest neighbors. Just store the values.
        datafeatures = features;
        datalabels = labels;
    }

    private double calculateDistance(double[] a, double[] b){
        double dist = 0;
        //sum squares for each feature;
        for(int i = 0; i < a.length; i ++)
        {
            dist += Math.pow((b[i] - a[i]), 2);
        }

        //take the root of the sum squares;

        dist = Math.sqrt(dist);
        return dist;
    }



    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        //Basically we just loop through all the features, and find the ones that are 'closest' in each distance.
        PriorityQueue<KNNPriQueue> queue = new PriorityQueue<KNNPriQueue>(_k, new KNNPriQueue(0,0));

        //Calculate the distance for each row.
        for(int j = 0; j < datafeatures.m_data.size(); j++){
            double[] row = datafeatures.m_data.get(j);

            if (features.length != row.length)
                throw new Exception("The value for prediction and the training set need to have the same feature space");

            double dist = calculateDistance(row, features);

            //check to see if this sum is closer than the current stored closest neighbors.
            if(queue.size() <_k){
                queue.add(new KNNPriQueue(dist, datalabels.row(j)[0]));
            } else if (queue.peek().weight > dist){
                queue.poll();
                queue.add(new KNNPriQueue(dist, datalabels.row(j)[0]));
            }
        }

        //Go find the value with the most instances in the queue.
        HashMap<Double, Double> counters = new HashMap<Double, Double>();

        double weight;

        //add up the number of each value
        for(KNNPriQueue q : queue){
            if (!_weighting)
                 weight = 1;
            else {
                //If it matches exactly, it's gonna be the same.
                if (q.weight == 0)
                    weight = Double.MAX_VALUE;
                //otherwise, weight it.
                else
                    weight = q.weight * (1.0 / Math.pow(q.weight, 2));
            }
            if (counters.containsKey(q.value)){
                counters.put(q.value, counters.get(q.value) + weight);
            }
            else
            {
                counters.put(q.value, weight);
            }
        }

        double curWinnerValue = Double.MIN_VALUE;
        double curWinner = Integer.MIN_VALUE;
        //find the most common.
        for (Double d : counters.keySet()){
            double challenger = counters.get(d);

            if (challenger > curWinner){
                curWinner = challenger;
                curWinnerValue = d;
            }
        }

        labels[0] = curWinnerValue;
    }
}
