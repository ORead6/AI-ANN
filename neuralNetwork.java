import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class neuralNetwork{

    public static int I_dim = 2;
    public static int H_dim = 2;
    public static int O_dim = 1;

    public static int epochCount = 1000000;
    public static Double learning_param = 0.1;

    public static double[][] weightToHid = new double[I_dim][H_dim];
    public static double[] weightToOut = new double[H_dim];
    public static double[] hidBias = new double[H_dim];
    public static double[] hidVals = new double[H_dim];
    public static double[] hidSelfWeight = new double[H_dim]; 
    public static double[] outSelfWeight = new double[O_dim]; 
    public static double outBias = Math.random() / H_dim;
    public static double outVal = 0.0;
    public static double[] outDelta = new double[O_dim];
    public static double[] hidDelta = new double[H_dim];

    public static double[][] data = getData();
    public static double[] desiredOut = {0.0, 1.0, 1.0, 0.0};

    public static double[][] getData(){
        double[][] theData = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        return theData;
    }

    public static void initWeights(){
        for (int i = 0; i < I_dim; i++){
            for (int x = 0; x < H_dim; x++){
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
        int minVal = 0;
        int maxVal = 1;

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
        int minVal = 0;
        int maxVal = 1;

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
        double[] inpVal = getData()[dataIndex];
        
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
            double[] errors = new double[getData().length];

            for (int i = 0; i < getData().length; i++){
                double[] thisPass = getData()[i];
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
        initWeights();
        plotGraph(feedForward(epochCount));
        System.out.println("Done");

    }
}