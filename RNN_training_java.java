package org.deeplearning4j.examples.feedforward.classification;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.text.DecimalFormat;

public class RNN_train {
    public static void main(String[] args) throws IOException, InterruptedException {
        //Member variables declaration
        int nEpochs = 200;
        int vectorSize = 5;

        //Load the training data:
        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0,",");

        featureReader.initialize(new NumberedFileInputSplit("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myInput_%d.csv", 0,2000));
        labelReader.initialize(new NumberedFileInputSplit("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myLabels_%d.csv", 0,2000));

        SequenceRecordReader ValfeatureReader = new CSVSequenceRecordReader(0, ",");
        SequenceRecordReader VallabelReader = new CSVSequenceRecordReader(0,",");

        ValfeatureReader.initialize(new NumberedFileInputSplit("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myInput_%d.csv",2001,3900));
        VallabelReader.initialize(new NumberedFileInputSplit("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myLabels_%d.csv",2001,3900));

        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(featureReader,labelReader,50,2,false,AlignmentMode.ALIGN_END);
        DataSetIterator trainIter2 = new SequenceRecordReaderDataSetIterator(featureReader,labelReader,1,2,false,AlignmentMode.ALIGN_END);
        DataSetIterator ValIter = new SequenceRecordReaderDataSetIterator(ValfeatureReader,VallabelReader,1,2,false,AlignmentMode.ALIGN_END);
        DataSetIterator ValIter2 = new SequenceRecordReaderDataSetIterator(ValfeatureReader,VallabelReader,1800,2,false,AlignmentMode.ALIGN_END);

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.RMSPROP)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(0.001)
            .list(2)
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(50)
                .activation("softsign").build())
            .layer(1, new RnnOutputLayer.Builder().activation("sigmoid")
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(50).nOut(2).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //training on Dataset , calculate score on training and validation error
        PrintWriter out2 = new PrintWriter("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\training_epochs.txt");
//        System.out.println("Starting training");
        for( int n=0; n<nEpochs; n++ ){
            model.fit(trainIter);
            trainIter.reset();
            ValIter2.reset();
            double train_score = model.score();
            double val_score = model.score(ValIter2.next(), false);
            trainIter2.reset();
            Evaluation evaluation2 = new Evaluation(2);
            while (trainIter2.hasNext()){
                DataSet ds_eval = trainIter2.next();
                INDArray features = ds_eval.getFeatureMatrix();
                INDArray lables = ds_eval.getLabels();
                INDArray inMask = ds_eval.getFeaturesMaskArray();
                INDArray outMask = ds_eval.getLabelsMaskArray();
                INDArray predicted = model.output(features,false,inMask,outMask);
                evaluation2.evalTimeSeries(lables,predicted,outMask);
            }

            ValIter.reset();
            Evaluation evaluation = new Evaluation(2);
            while (ValIter.hasNext()){
                DataSet ds_eval = ValIter.next();
                INDArray features = ds_eval.getFeatureMatrix();
                INDArray lables = ds_eval.getLabels();
                INDArray inMask = ds_eval.getFeaturesMaskArray();
                INDArray outMask = ds_eval.getLabelsMaskArray();
                INDArray predicted = model.output(features,false,inMask,outMask);
                evaluation.evalTimeSeries(lables,predicted,outMask);
            }
            DecimalFormat f = new DecimalFormat("##.0000");
            System.out.println("Epoch " + n + " Train Score,Val score,Train Accuracy,Val Accuracy" +" : "+ f.format(train_score)+" , "+ f.format(val_score)+" , " + f.format(evaluation2.accuracy()) + " , " + f.format(evaluation.accuracy()));

            //write network parameters and configurations
            try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Model_Configs\\coefficients" +n+ ".bin"))))
            {
                Nd4j.write(model.params(),dos);
            }
            //Write the network configuration:
            FileUtils.write(new File("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Model_Configs\\conf" +n+".json"), model.getLayerWiseConfigurations().toJson());
        }
        out2.close();
    }
}
