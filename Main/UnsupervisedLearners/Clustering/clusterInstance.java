package Main.UnsupervisedLearners.Clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


/**
 * Created by joseph on 3/28/16.
 */
public class clusterInstance {


    public double[] centroid;
    public ArrayList<Integer> instanceLocations;
    public ArrayList<double[]> instances;

    public clusterInstance(double[] centroid) {
        //this.centroid = centroid;

        instanceLocations = new ArrayList<Integer>();
        instances = new ArrayList<double[]>();

        this.centroid = new double[centroid.length];

        for (int i = 0; i < this.centroid.length; i++) {
            this.centroid[i] = centroid[i];
        }
    }

    public clusterInstance clone() {
        return new clusterInstance(this.centroid.clone());
    }


    public boolean equals(Object o) {
        if (!(o instanceof clusterInstance)) {
            return false;
        }
        if (o == this)
            return true;


        clusterInstance toCompare = (clusterInstance) o;

        if (toCompare.centroid.length != this.centroid.length)
            return false;

        return Arrays.equals(toCompare.centroid, this.centroid);
    }

    public void addInstance(int location, double[] instance) {
        instances.add(instance);
        instanceLocations.add(location);
    }

    public void recalculateCentroid(boolean[] NominalValues) {
        //go through and calculate the mean of each item.

        for (int i = 0; i < centroid.length; i++) {
            if (!NominalValues[i]) {
                centroid[i] = 0;
                int count = 0;

                for (double[] d : instances) {
                    if (d[i] == Double.MAX_VALUE)
                        continue;

                    centroid[i] += d[i];
                    count++;
                }
                //If all were unknown, leave it unknown.
                if (count == 0) {
                    centroid[i] = Double.MAX_VALUE;
                }
                else
                    centroid[i] = centroid[i] / count;
            }
            //If it's a nominal value, find the majority case.
            else {
                HashMap<Double, MutableInt> counts = new HashMap<Double, MutableInt>();

                for (double[] d : instances) {
                    if (d[i] == Double.MAX_VALUE)
                        continue;

                    if (counts.containsKey(d[i]))
                        counts.get(d[i]).increment();
                    else {
                        counts.put(d[i], new MutableInt());
                    }
                }

                double curWinner = -1;
                int curBest = Integer.MIN_VALUE;
                for (Double d : counts.keySet()){

                    int challenger = counts.get(d).get();

                    if (challenger > curBest){
                        curWinner = d;
                        curBest = challenger;
                    }
                }

                centroid[i] = curWinner;
            }
        }
    }


    public int hashCode() {
        int hash = 0;

        for (int i = 0; i < centroid.length; i++) {
            hash = (hash * (int) centroid[i]);
        }
        hash = hash * 40362779;

        return hash;
    }

    class MutableInt {
        int value = 1;

        public void increment() {
            value++;
        }

        public int get() {
            return value;
        }
    }
}
