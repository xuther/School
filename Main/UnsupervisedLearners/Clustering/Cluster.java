package Main.UnsupervisedLearners.Clustering;

import Main.Matrix;
import Main.SupervisedLearners.SupervisedLearner;

import java.util.ArrayList;
import java.util.Random;
/**
 * Created by joseph on 3/28/16.
 */
public class Cluster extends SupervisedLearner {

    public int k;

    private ArrayList<clusterInstance> clusters;
    private ArrayList<clusterInstance> oldClusters;
    private Random r;

    private Matrix features;

    private boolean[] nominalValues;

    public Cluster(int k){
        super();
        this.k = k;
        clusters = new ArrayList<clusterInstance>();
        oldClusters = new ArrayList<clusterInstance>();
        r = new Random();
    }


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        ArrayList<Integer> numbers = new ArrayList<Integer>();
        int next;

        this.features = features;

        populateNominal(features);

        if (features.rows() < k)
            throw new Exception("There wasn't enough data in the set to justify the number of desired clusters.");

        for (int i = 0; i < features.rows(); i++){
            numbers.add(i);
        }

        //inital cluster centroids.
        for (int i = 0; i < k; i++ ){
            next = r.nextInt(numbers.size());
            next = numbers.get(next);
            //next = numbers.get(i);
            clusters.add(new clusterInstance(features.row(next)));
            numbers.remove(next);
        }

        //go for each of our items. group them, and then recalculate the centroid
        while(areDifferences()) {

            oldClusters = clusters;
            clusters = new ArrayList<clusterInstance>();

            //copy the centroids from old clusters to new clusters
            for (int i = 0; i < oldClusters.size(); i++)
            {
                clusters.add(oldClusters.get(i).clone());
            }

            while (numbers.size() > 0) {
                next = numbers.get(0);

                numbers.remove(0);

                int nearest = calculateNearest(features.row(next), clusters);
                clusters.get(nearest).addInstance(next, features.row(next));
            }

            for (int i = 0; i < features.rows(); i++){
                numbers.add(i);
            }

            double SSE = 0;
            for(clusterInstance i : clusters){
                SSE += calculateSSE(i);
                i.recalculateCentroid(nominalValues);
            }
            System.out.println("SSE: " + SSE);
        }
        printClusters();
    }

    private void printClusters()
    {
        double avgSillhouette = 0;
        double totalSSE = 0;
        for (int i = 0; i < k; i++)
        {
            clusterInstance current = clusters.get(i);
            double Sillhouette = calculateAverageSillhouette(current);
            double SSE = calculateSSE(current);

            System.out.println("Cluster: " + i +  "       SSE: " + SSE + "      Sillhouette: " + Sillhouette +
                    "       Instances: " + current.instances.size());
            System.out.print("Centroid: " );
            for (int j = 0; j < current.centroid.length; j ++)
            {
                if (nominalValues[j])
                    System.out.print(features.m_enum_to_str.get(j).get((int)current.centroid[j]) + ", ");
                else if (current.centroid[j] == Double.MAX_VALUE)
                    System.out.print( "?, ");
                else
                    System.out.print(current.centroid[j] + ", ");
            }
            System.out.println("");
            System.out.println("");

            avgSillhouette += Sillhouette;
            totalSSE += SSE;
        }
        avgSillhouette = avgSillhouette/k;
        System.out.println("Total SSE: " + totalSSE);
        System.out.println("Average sillhouette: " + avgSillhouette);
    }

    private double calculateAverageSillhouette(clusterInstance c)
    {
        double avgSillhouette = 0;

        for (double[] current : c.instances)
        {
            //find the next nearest cluster;
            avgSillhouette += calculateSillhouette(current, c, findNextNearest(current, c));
        }
        avgSillhouette = avgSillhouette/c.instances.size();

        return avgSillhouette;
    }


    private clusterInstance findNextNearest(double[] toFind, clusterInstance assigned){
        ArrayList<clusterInstance> modifiedList = new ArrayList<clusterInstance>(clusters);

        modifiedList.remove(assigned);

        return modifiedList.get(calculateNearest(toFind, modifiedList));
    }


    private double calculateSSE(clusterInstance c)
    {
        double SSE = 0;
        for (double[] d : c.instances)
        {
            SSE += (Math.pow(calculateDist(d,c.centroid), 2));
        }
        return SSE;
    }


    //Check if there have been changes in the centroids
    private boolean areDifferences()
    {
        if (clusters.size() != oldClusters.size())
            return true;

       for (int i = 0; i < clusters.size(); i++)
       {
           if (!clusters.get(i).equals(oldClusters.get(i))){
               return true;
           }
       }
        return false;
    }

    private double calculateSillhouette(double[] instance, clusterInstance member, clusterInstance nextChoice)
    {
        double aOfI = 0;
        //calculate similarity to current cluster
        for (int i = 0; i < member.instances.size(); i++)
        {
            if (instance == member.instances.get(i))
                continue;

            aOfI += calculateDist(instance, member.instances.get(i));
        }
        aOfI = aOfI/(member.instances.size() -1);

        double bOfI = 0;
        //calculate dissimilary to the next best cluster
        for (int i = 0; i < nextChoice.instances.size(); i++)
        {
            if (instance == nextChoice.instances.get(i))
                continue;

            bOfI += calculateDist(instance, nextChoice.instances.get(i));
        }
        bOfI = bOfI/(nextChoice.instances.size() -1);

        //Find the max of aOfI or bOfI - if they're equal it doesn't matter - it'll come out to zero anyway.
        double denominator = .0001;
        if (aOfI > bOfI)
            denominator = aOfI;
        else
            denominator = bOfI;

        if (denominator == 0)
            return 0;

        //return the sillhouette.
        return ((bOfI - aOfI)/denominator);
    }



    private void populateNominal(Matrix f){
        nominalValues = new boolean[f.cols()];

        for (int i = 0; i < f.cols(); i++)
        {
            if (f.m_enum_to_str.get(i).size() != 0)
                nominalValues[i] = true;
            else
                nominalValues[i] = false;
        }

    }


    private int calculateNearest(double[] row, ArrayList<clusterInstance> clusters){

        int curWinner = -1;
        double curBest = Double.MAX_VALUE;
        for (int i = 0; i < clusters.size(); i++)
        {
            double[] challengerRow = clusters.get(i).centroid;
            double challenger = calculateDist(challengerRow, row);

            if (challenger < curBest)
            {
                curWinner = i;
                curBest = challenger;
            }
        }

        return curWinner;
    }

    public double calculateDist(double[] a, double[] b){

        double dist = 0;
        //sum squares for each feature;
        for(int i = 0; i < a.length; i ++)
        {
            //IS THIS REALLY UNKNOWN VALUE?
            if (a[i] == Double.MAX_VALUE || b[i] == Double.MAX_VALUE)
            {
                dist += 1;
            }
            else if (nominalValues[i]){
                if (a[i] != b[i])
                {
                    dist += 1;
                }
            }
            else {
                dist += Math.pow((b[i] - a[i]), 2);
            }
        }
        //take the root of the sum squares;
        dist = Math.sqrt(dist);
        return dist;
    }


    //We don't predict.
    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        labels[0] = 0;
    }

}
