package Main.SupervisedLearners.KNN;

import java.util.Comparator;
/**
 * Created by joseph on 3/18/16.
 */
public class KNNPriQueue implements Comparator<KNNPriQueue>, Comparable<KNNPriQueue> {

    //Weight is the place 'ranking' in the priority queue. Ranked on from highest to lowest (the head of the priority
    //queue will be the item with the largest weight.)
    public double weight;
    public double value;


    public KNNPriQueue(double weight, double value)
    {
        this.weight = weight;
        this.value = value;
    }


    @Override
    public int compareTo(KNNPriQueue o) {
        if (o.weight > this.weight){
            return -1;
        }
        else if (o.weight < this.weight){
            return 1;
        }
        else
            return 0;
    }

    //This seems backwards only because we're looking to use it in a Pri queue, and Java's pri queue
    //spits out the lowest value, and we want the largest.
    @Override
    public int compare(KNNPriQueue o1, KNNPriQueue o2) {
        if (o1.weight > o2.weight){
            return -1;
        }
        else if (o1.weight < o2.weight){
            return 1;
        }
        else
            return 0;
    }
}

