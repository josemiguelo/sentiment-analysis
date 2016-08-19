/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redesNeuronales;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Jose
 */
public class RedNeuronal {

    public void training(String paramNN) throws FileNotFoundException, IOException, Exception {
        //enrenamiento de la red neuronal con parámetros 
        FileReader trainReader = new FileReader("src/archivos/outputTrain.arff");
        //Instancias
        Instances trainInstance = new Instances(trainReader);
        trainInstance.setClassIndex(trainInstance.numAttributes() - 1);

        //Construir la red
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setOptions(Utils.splitOptions(paramNN)); //fijar parámetros de multilayerPerceptron
        mlp.buildClassifier(trainInstance);//construir clasificador
        
        //Bayes
        NaiveBayes naiveBayes = new NaiveBayes();

        //Guardar el perceptrón multicapa - generar el archivo
        Debug.saveToFile("TrainMLP.train", mlp);

        //Evaluar el entrenamiento
        SerializedClassifier sc = new SerializedClassifier();
        sc.setModelFile(new File("TrainMLP.train"));
        Evaluation evaluateTrain = new Evaluation(trainInstance);
        evaluateTrain.evaluateModel(mlp, trainInstance);//evaluación del modelo
        System.err.println(evaluateTrain.toSummaryString("******Resultados*********", false));
        System.err.println(evaluateTrain.toMatrixString("******Matriz de confusión****"));

        trainReader.close();
    }

    public void trainingNB (String paramNN) throws FileNotFoundException, IOException, Exception {
        //enrenamiento de la red neuronal con parámetros 
        FileReader trainReader = new FileReader("src/archivos/outputTrain.arff");
        //Instancias
        Instances trainInstance = new Instances(trainReader);
        trainInstance.setClassIndex(trainInstance.numAttributes() - 1);

        //Construir la red
        NaiveBayes naiveBayes = new NaiveBayes();
        //naiveBayes.setOptions(Utils.splitOptions(paramNN)); //fijar parámetros de multilayerPerceptron
        naiveBayes.buildClassifier(trainInstance);//construir clasificador

        //Guardar el perceptrón multicapa - generar el archivo
        Debug.saveToFile("TrainBayes.train", naiveBayes);

        //Evaluar el entrenamiento
        SerializedClassifier sc = new SerializedClassifier();
        sc.setModelFile(new File("TrainBayes.train"));
        Evaluation evaluateTrain = new Evaluation(trainInstance);
        evaluateTrain.evaluateModel(naiveBayes, trainInstance);//evaluación del modelo
        System.err.println(evaluateTrain.toSummaryString("******Resultados*********", false));
        System.err.println(evaluateTrain.toMatrixString("******Matriz de confusión****"));

        trainReader.close();
    }
    
    public void trainingJ48 (String paramNN) throws FileNotFoundException, IOException, Exception {
        //enrenamiento de la red neuronal con parámetros 
        FileReader trainReader = new FileReader("src/archivos/outputTrain.arff");
        //Instancias
        Instances trainInstance = new Instances(trainReader);
        trainInstance.setClassIndex(trainInstance.numAttributes() - 1);

        //Construir la red
        J48 j48 = new J48();
        //j48.setOptions(Utils.splitOptions(paramNN)); //fijar parámetros de multilayerPerceptron
        j48.buildClassifier(trainInstance);//construir clasificador

        //Guardar el perceptrón multicapa - generar el archivo
        Debug.saveToFile("TrainBayes.train", j48);

        //Evaluar el entrenamiento
        SerializedClassifier sc = new SerializedClassifier();
        sc.setModelFile(new File("TrainJ48.train"));
        Evaluation evaluateTrain = new Evaluation(trainInstance);
        evaluateTrain.evaluateModel(j48, trainInstance);//evaluación del modelo
        System.err.println(evaluateTrain.toSummaryString("******Resultados*********", false));
        System.err.println(evaluateTrain.toMatrixString("******Matriz de confusión****"));

        trainReader.close();
    }

    public void testing() throws FileNotFoundException, IOException, Exception {
        ArrayList<Prediction> predicciones = new ArrayList<>();
        ArrayList<String> cabecera = new ArrayList<String>();
        ArrayList<String> datos = new ArrayList<String>();
        String lineaActual;
        int i = 0;
        double prediccion;
        FileReader testReader = new FileReader("src/archivos/outputSinClasificar.arff");
        Instances testInstance = new Instances(testReader);
        testInstance.setClassIndex(testInstance.numAttributes() - 1);

        weka.classifiers.evaluation.Evaluation evaluateTest = new weka.classifiers.evaluation.Evaluation(testInstance);

        SerializedClassifier sc = new SerializedClassifier();
        sc.setModelFile(new File("TrainMLP.train"));

        //Cargar clasificardor desde el .train
        Classifier mlp = sc.getCurrentModel();

        evaluateTest.evaluateModel(mlp, testInstance);//devuelve un arreglo de dobles
        predicciones = evaluateTest.predictions();
        Scanner scanner = new Scanner(new File("src/archivos/outputSinClasificar.arff"));
        while (scanner.hasNextLine()) {
            lineaActual = scanner.nextLine();
            if (lineaActual.matches("^[@%].*$") || lineaActual.matches("^\\s*$")) {
                cabecera.add(lineaActual);
            } else {
                datos.add(lineaActual);
            }
        }
        Iterator<Prediction> prediccionesIterator = predicciones.iterator();
        while (prediccionesIterator.hasNext()) {
            prediccion = prediccionesIterator.next().predicted();
            //System.out.println(prediccion);
            switch ((int) prediccion) {
                case 0:
                    datos.set(i, datos.get(i).replaceAll("\\?", "positive"));
                    //System.out.println("brickface");
                    break;
                case 1:
                    datos.set(i, datos.get(i).replaceAll("\\?", "negative"));
                    //System.out.println("sky");
                    break;
                case 2:
                    datos.set(i, datos.get(i).replaceAll("\\?", "neutral"));
                    //System.out.println("foliage");
                    break;
                default:
                    System.out.println("unknown");
                    break;
            }
            //System.out.println(datos.get(i));
            i++;
        }
        Iterator<String> imprimirCabecera = cabecera.iterator();
        while (imprimirCabecera.hasNext()) {
            System.out.println(imprimirCabecera.next());
        }
        Iterator<String> imprimirDatos = datos.iterator();
        while (imprimirDatos.hasNext()) {
            System.out.println(imprimirDatos.next());
        }
        FileWriter writer = new FileWriter("src/archivos/outputClasificado.arff");
        for (String str : cabecera) {
            writer.write(str+"\n");
        }
        for (String str : datos) {
            writer.write(str+"\n");
        }
        writer.close();
    }

    public void validate() throws FileNotFoundException, IOException, Exception {
        FileReader testReader = new FileReader("src/archivos/sickValidate.arff");
        Instances testInstance = new Instances(testReader);
        testInstance.setClassIndex(testInstance.numAttributes() - 1);

        weka.classifiers.evaluation.Evaluation evaluateTest = new weka.classifiers.evaluation.Evaluation(testInstance);

        SerializedClassifier sc = new SerializedClassifier();
        sc.setModelFile(new File("TrainMLP.train"));

        //Cargar clasificardor desde el .train
        Classifier mlp = sc.getCurrentModel();

        evaluateTest.evaluateModel(mlp, testInstance);//devuelve un arreglo de dobles

        //System.err.println(evaluateTest.toSummaryString("******Resultados*********", false));
        //System.err.println(evaluateTest.toMatrixString("******Matriz de confusión****"));
    }
}
