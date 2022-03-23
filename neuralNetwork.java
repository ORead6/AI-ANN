import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class neuralNetwork{

    public static int I_dim = 8;
    public static int H_dim = 5;
    public static int O_dim = 1;

    public static int epochCount = 100000;
    public static Double learning_param = 0.2;

    public static double minVal = 3.694; //Change this depending on data set
    public static double maxVal = 448.1; //Change this depending on data set

    public static double[][] weightToHid = new double[H_dim][I_dim];
    public static double[] weightToOut = new double[H_dim];
    public static double[] hidBias = new double[H_dim];
    public static double[] hidVals = new double[H_dim];
    public static double[] hidSelfWeight = new double[H_dim]; 
    public static double[] outSelfWeight = new double[O_dim]; 
    public static double outBias = Math.random() / H_dim;
    public static double outVal = 0.0;
    public static double[] outDelta = new double[O_dim];
    public static double[] hidDelta = new double[H_dim];

    //XOR Testing
    //public static double[][] data = getData();
    //public static double[] desiredOut = {0.0, 1.0, 1.0, 0.0};

    public static double[][] getData(){
        //XOR
        //double[][] theData = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        try {
            ArrayList<String[]> theData1 = new ArrayList<String[]>();
            BufferedReader csvReader = new BufferedReader(new FileReader("DataSet.csv"));
            String row;
            int x = 0;
            while ((row = csvReader.readLine()) != null) {
                if (x > 1){
                    String[] data = row.split(",");
                    theData1.add(Arrays.copyOfRange(data, 1, data.length));
                } else {x++;}
            }

            csvReader.close();

            double[][] theData = new double[theData1.size()][8];

            int size2 = theData1.size();
            for (int i = 0; i < size2; i++){
                int size3 = theData1.get(i).length;
                double[] temp = new double[size3];
                for (int y = 0; y < size3; y++){
                    if (theData1.get(i)[y] != ""){
                        temp[y] = Double.parseDouble(theData1.get(i)[y]);
                    }
                }

                theData[i] = temp;
            }

            return theData;


        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return data;

    }

    public static double[] getDesiredData(){
        //XOR
        //double[][] theData = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        try {
            ArrayList<String> desiredOut1 = new ArrayList<String>();
            BufferedReader csvReader = new BufferedReader(new FileReader("DataSet.csv"));
            String row;
            int x = 0;
            while ((row = csvReader.readLine()) != null) {
                if (x > 1){
                    String[] data = row.split(",");
                    desiredOut1.add(data[3]);
                } else {x++;}
            }

            csvReader.close();

            double[] desiredOut = new double[desiredOut1.size()];

            int size = desiredOut1.size();
            for (int i = 0; i < size; i++){
                desiredOut[i] = Double.parseDouble(desiredOut1.get(i));
            }

            return desiredOut;


        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return desiredOut;

    }

    public static double[][] data = getData();
    public static double[] desiredOut = getDesiredData();

    public static void initWeights(){
        for (int i = 0; i < H_dim; i++){
            for (int x = 0; x < I_dim; x++){
                weightToHid[i][x] = Math.random() / I_dim;
            }
        }

        for (int i = 0; i < H_dim; i++){
            weightToOut[i] = Math.random() / H_dim;
            hidBias[i] = Math.random() / H_dim;
            hidVals[i] = Math.random() / I_dim;
            hidSelfWeight[i] = Math.random() / I_dim;
            hidDelta[i] = 0.0;
        }

        outSelfWeight[0] = Math.random() / H_dim;
        outDelta[0] = 0.0;
    }

    public static double[] normaliseData(double[] data, String thisCase){

        if (thisCase == "pre"){
            double[] answer = new double[data.length];
            int size = answer.length;
            for (int i = 0; i < size; i++){
                answer[i] = (data[i] - minVal) / (maxVal - minVal);
            }
            return answer;

        } else {
            double[] answer = new double[data.length];
            int size = answer.length;
            for (int i = 0; i < size; i++){
                answer[i] = (data[i] * (maxVal - minVal)) + minVal;
            }
            return answer;
        }

        
    }

    public static double normaliseSingle(double data, String thisCase){

        if (thisCase == "pre"){
            double answer;
            answer = (data - minVal) / (maxVal - minVal);
            return answer;

        } else {
            double answer;
            answer = (data * (maxVal - minVal)) + minVal;
            return answer;
        }
    }

    public static double wSum(double[] m1, double[] m2){
        double sum = 0.0;
        int size = m1.length;
        for (int i = 0; i < size; i++){
            sum += m1[i] * m2[i];
            
        }
        return sum;
    }

    public static double errorFunc(double predicted, double real){
        return Math.pow((predicted - real), 2);
    }

    public static double activation(double x){
        //Sigmoid
        double result = 1 / (1 + Math.exp(-x));
        return result;
    }

    public static double getOverall(double [] thisData){
        double sum = 0.0;
        int size = thisData.length;
        for (int i = 0; i < size; i++){
            sum += thisData[i];
        }
        return sum;
    }

    public static void backProp(int dataIndex, double observed){
        double desiredOutVal = normaliseSingle(observed, "pre");
        double[] inpVal = data[dataIndex];
        
        for (int i = 0; i < O_dim; i++){
            outDelta[i] = (desiredOutVal - outVal) * derivative(outVal);
            outBias += (learning_param * outDelta[i]);
            for (int x = 0; x < H_dim; x++){
                weightToOut[x] += (learning_param * outDelta[i] * hidVals[x]);
                hidDelta[x] = (weightToOut[x] * outDelta[i]) * derivative(hidVals[x]);
                hidBias[x] += (learning_param * hidDelta[x]);
                for (int j = 0; j < I_dim; j++){
                    weightToHid[x][j] += (learning_param * hidDelta[x] * normaliseSingle(inpVal[j], "pre"));
                }
            }
        } 

    }

    private static double derivative(double x) {
        return (x * (1 - x));
    }

    public static double[] feedForward(int epochs){
        double[] epochErrors = new double[epochs];
        for (int j = 0; j < epochs; j++){
            double[] errors = new double[data.length];

            for (int i = 0; i < data.length; i++){
                double[] thisPass = data[i];
                for (int x = 0; x < H_dim; x++){
                    double[] weights = weightToHid[x];

                    double[] norm = normaliseData(thisPass, "pre"); 
                    double wS = wSum(weights, norm) + (hidSelfWeight[x] * hidVals[x]);

                    hidVals[x] = activation(wS);
                }

                for (int x = 0; x < O_dim; x++){
                    double wS = wSum(weightToOut, hidVals) + (outSelfWeight[x] * outVal);
                    outVal = activation(wS);

                    double error = errorFunc(outVal, normaliseSingle(desiredOut[i], "pre"));
                    
                    //String results = String.format("Expected: %f             Got: %f", desiredOut[i], outVal);
                    //System.out.println(results);
                    errors[i] = error;
                }

                backProp(i, desiredOut[i]);

            }
            epochErrors[j] = getOverall(errors);
        } 

        return(epochErrors);
    }

    public static void plotGraph(double[] errorGraph) throws IOException{
        new FileWriter("errorData.txt").close();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("errorData.txt"))){
            for (double line: errorGraph){
                writer.write (line + "\n");
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace ();
        }
    }

    public static void main(String[] args) throws IOException{
        System.out.println(data.length);
        System.out.print(desiredOut.length);
        initWeights();
        plotGraph(feedForward(epochCount));
        System.out.println("Done");

    }
}