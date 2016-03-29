package Main.SupervisedLearners.KNN;

import Main.Matrix;
import Main.SupervisedLearners.SupervisedLearner;

import java.util.DoubleSummaryStatistics;
import java.util.PriorityQueue;
import java.util.HashMap;
/**
 * Created by joseph on 3/18/16.
 */
public class KNN extends SupervisedLearner {

    private Matrix datafeatures;
    private Matrix datalabels;
    boolean[] nominalFeatures;
    private boolean _weighting;
    private boolean _regression;
    private boolean _manhatten = true;
    private int _k;

    public KNN(boolean weighting, boolean regression, int k)
    {
        _weighting = weighting;
        _regression = regression;


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

        nominalFeatures=  new boolean[features.row(0).length];

        for (int i = 0; i < features.row(0).length; i++){
            nominalFeatures[i] = (features.m_str_to_enum.get(i).size() > 0);
        }

    }

    private double calculateDistance(double[] a, double[] b, boolean[] nominal) {
        double dist = 0;
        int unknowns = 0;
        if (_manhatten) {
            for (int i = 0; i < a.length; i++) {
                //if unknown, tag it and move on.
                if (a[i] == Double.MAX_VALUE || b[i] == Double.MAX_VALUE){
                    unknowns++;
                    continue;
                }
                if(!nominal[i]) {
                    //we need to check for unknown values
                    dist += Math.abs((b[i] - a[i]));
                }
                else {
                    //we need to check for unknown

                    //if they're not equal, distance of 1, otherwise let it be.
                    if (a != b){
                        dist += 1;
                    }
                }
            }

            //For each unknown value, take the average distance of everything else and add it to the total distance.
            //Basically we assume that the unknown items are about as far away - on average - as everything else.
            //take the root of the sum squares;
            if (unknowns != 0)
            {
                double avgDist = dist/(a.length-unknowns);
                for (int i = 0; i < unknowns; i++)
                {
                    dist += avgDist;
                }
            }
            dist = Math.sqrt(dist);

            return dist;
        }
        //sum squares for each feature;
        for (int i = 0; i < a.length; i++) {
            //if unknown, tag it and move on.
            if (a[i] == Double.MAX_VALUE || b[i] == Double.MAX_VALUE){
                unknowns++;
                continue;
            }
            if(!nominal[i]) {
                //we need to check for unknown values
                dist += Math.pow((b[i] - a[i]), 2);
            }
            else {
                //we need to check for unknown

                //if they're not equal, distance of 1, otherwise let it be.
                if (a != b){
                    dist += 1;
                }
            }
        }

        //For each unknown value, take the average distance of everything else and add it to the total distance.
        //Basically we assume that the unknown items are about as far away - on average - as everything else.
        //take the root of the sum squares;
        if (unknowns != 0)
        {
            double avgSqrdDist = dist/(a.length-unknowns);
            for (int i = 0; i < unknowns; i++)
            {
                dist += avgSqrdDist;
            }
        }
        dist = Math.sqrt(dist);

        return dist;
    }

    public void setK(int k)
    {
        this._k = k;
    }
    public int getk() {
        return _k;
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

            double dist = calculateDistance(row, features, nominalFeatures);

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
        if (!_regression) {
            for (KNNPriQueue q : queue) {
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
                if (counters.containsKey(q.value)) {
                    counters.put(q.value, counters.get(q.value) + weight);
                } else {
                    counters.put(q.value, weight);
                }
            }
        }
        //If it's regression we can just set labels as the mean of k nearest neighbors (with or without weighting
        //and return.
        else {
            double sumTop = 0.0;
            double sumBot = 0.0;
            double value = 0.0;
            if (_weighting) {
                for (KNNPriQueue q : queue) {
                    sumTop += (q.value * (1.0 / Math.pow(q.weight, 2)));
                    sumBot += (1.0 / Math.pow(q.weight, 2));
                }
            } else{
                for (KNNPriQueue q : queue) {
                    sumTop += q.value;
                    sumBot = sumBot + 1;
                }
            }

            value = sumTop/sumBot;
            labels[0] = value;
            return;
        }

        double curWinnerValue = Double.MIN_VALUE;
        double curWinner = Integer.MIN_VALUE;
        //find the most common.
        for (Double d : counters.keySet()) {
            double challenger = counters.get(d);

            if (challenger > curWinner) {
                curWinner = challenger;
                curWinnerValue = d;
            }
        }

        labels[0] = curWinnerValue;

    }
}
