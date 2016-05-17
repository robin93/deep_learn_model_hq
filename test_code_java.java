package org.deeplearning4j.examples.feedforward.classification;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;


import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
/**
 * Created by Administrator on 5/16/2016.
 */
public class MLPClassifierLinear_test {
    public static void main(String[] args) throws Exception {

        //Load the test data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\walk_forward_learn_on_16May\\95_to_98\\test_data_java.csv")));
//        int val_set_size = 7285;
//        DataSetIterator ValIter = new RecordReaderDataSetIterator(rrEval,val_set_size,38,2);
//        INDArray features_val;
//        INDArray lables_val;
//        DataSet t1 = ValIter.next();
//        features_val = t1.getFeatureMatrix();
//        lables_val = t1.getLabels();
//        DataSet ValData = new org.nd4j.linalg.dataset.DataSet(features_val,lables_val);


        //Load parameters from disk:
        INDArray newParams;
        try (DataInputStream dis = new DataInputStream(new FileInputStream("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\walk_forward_learn_on_16May\\95_to_98\\coefficients123.bin"))) {
            newParams = Nd4j.read(dis);
        }

        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\walk_forward_learn_on_16May\\95_to_98\\conf123.json")));

        MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
        model.init();
        model.setParameters(newParams);

        PrintWriter out = new PrintWriter("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\walk_forward_learn_on_16May\\95_to_98\\Prediction_95_98_walk_forward.txt");
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, 1, 39, 2);
        int count = 0;
        while (testIter.hasNext()) {
            DataSet ds_predict = testIter.next();
            INDArray features = ds_predict.getFeatureMatrix();
            INDArray predicted = model.output(features, false);
            int[] prediction_value = model.predict(features);
            out.println(predicted + "," + prediction_value[0]);
            System.out.println("count" + count + "--"+predicted + "," + prediction_value[0]);
            count = count + 1;
//            System.out.println(prediction_value[0])
        }
    }
}
