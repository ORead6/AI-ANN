import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.IntStream;

public class neuralNetwork{

    // Initialising Variables

    public static int I_dim = 8;
    public static int H_dim = 8; //5;
    public static int O_dim = 1;

    public static int epochCount = 4;
    public static Double learning_param = 0.1;

    public static double minVal = 3.694; //0; Change this depending on data set
    public static double maxVal = 448.1; //1; Change this depending on data set

    public static double[][] weightToHid = new double[H_dim][I_dim];
    public static double[] weightToOut = new double[H_dim];
    public static double[] hidBias = new double[H_dim];
    public static double[] hidVals = new double[H_dim];
    public static double[] hidSelfWeight = new double[H_dim]; 
    public static double[] outSelfWeight = new double[O_dim]; 
    public static double outBias = 0;
    public static double outVal = 0.0;
    public static double[] outDelta = new double[O_dim];
    public static double[] hidDelta = new double[H_dim];

    public static ArrayList<Double> dotExpected = new ArrayList<Double>();
    public static ArrayList<Double> dotGot = new ArrayList<Double>();

    //XOR Testing
    //public static double[][] data = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}; //getData("data"); 
    public static double[] desiredOut = getDesiredData("training"); //{0.0, 1.0, 1.0, 0.0};

    public static double[][] data = getData("training");
    public static double[][] validation = getData("validation");
    public static double[][] test = getData("test");

    // Used to convert 2D ArrayLists into 2D Double

    public static double[][] convertArrayList(ArrayList<String[]> theList){
        double[][] result = new double[theList.size()][8];
        int size = theList.size();
        for (int i = 0; i < size; i++){
            int size3 = theList.get(i).length;
            double[] temp = new double[size3];
            for (int y = 0; y < size3; y++){
                if (theList.get(i)[y] != ""){
                    temp[y] = Double.parseDouble(theList.get(i)[y]);
                }
            }

            result[i] = temp;
        }
        return result;
    }

    //Used to convert 1D ArrayLists into 1D Double Lists

    public static double[] convert1DArrayList(ArrayList<String> theList){
        double[] result = new double[theList.size()];
        int size = theList.size();
        for (int i = 0; i < size; i++){
            result[i] = Double.parseDouble(theList.get(i));
        }
        return result;
    }

    //Reads data from csv and puts into array for later use

    public static double[][] getData(String option){
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

            theData1.remove(theData1.size() - 1);

            csvReader.close();
            double i1 = Math.round(theData1.size() * 0.6);
            double i2 = i1 + Math.round(theData1.size() * 0.2);

            ArrayList<String[]> trainingSet1 = new ArrayList<String[]>();
            ArrayList<String[]> ValidationSet1 = new ArrayList<String[]>();
            ArrayList<String[]> TestSet1 = new ArrayList<String[]>();

            for (int i = 0; i < theData1.size(); i++){
                if (i < i1){
                    trainingSet1.add(theData1.get(i));
                }
                if (i1 <= i && i < i2){
                    ValidationSet1.add(theData1.get(i));
                }
                if (i >= i2){
                    TestSet1.add(theData1.get(i));
                }
            }

            double[][] trainingSet = convertArrayList(trainingSet1);
            double[][] validationSet = convertArrayList(ValidationSet1);
            double[][] testSet = convertArrayList(TestSet1);


            if (option.equalsIgnoreCase("training")){
                return trainingSet;
            }
            if (option.equalsIgnoreCase("validation")){
                return validationSet;
            }
            if (option.equalsIgnoreCase("test")){
                return testSet;
            }



        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return data;

    }

    //Reads data from csv and puts into array for later use
    public static double[] getDesiredData(String option){
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
                    desiredOut1.add(data[4]);
                } else {x++;}
            }

            desiredOut1.remove(0);

            csvReader.close();

            ArrayList<String> trainingOut = new ArrayList<String>();
            ArrayList<String> ValidationOut = new ArrayList<String>();
            ArrayList<String> TestOut = new ArrayList<String>();

            double i1 = Math.round(desiredOut1.size() * 0.6);
            double i2 = i1 + Math.round(desiredOut1.size() * 0.2);

            for (int i = 0; i < desiredOut1.size(); i++){
                if (i < i1){
                    trainingOut.add(desiredOut1.get(i));
                }
                if (i1 <= i && i < i2){
                    ValidationOut.add(desiredOut1.get(i));
                }
                if (i >= i2){
                    TestOut.add(desiredOut1.get(i));
                }
            }

            if (option.equalsIgnoreCase("training")){
                return convert1DArrayList(trainingOut);
            }
            if (option.equalsIgnoreCase("validation")){
                return convert1DArrayList(ValidationOut);
            }
            if (option.equalsIgnoreCase("test")){
                return convert1DArrayList(TestOut);
            }


        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return desiredOut;

    }

    //public static double[][] data = getData();
    //public static double[] desiredOut = getDesiredData();

    //Initialises Weight values at first run

    public static void initWeights(){
        int max = 1;
        int min = -1;
        for (int i = 0; i < H_dim; i++){
            for (int x = 0; x < I_dim; x++){
                weightToHid[i][x] = Math.random() / I_dim;
            }
        }

        for (int i = 0; i < H_dim; i++){
            weightToOut[i] = ((Math.random() * (max - min)) + min) / H_dim;
            hidBias[i] = 0;
            hidVals[i] = ((Math.random() * (max - min)) + min) / I_dim;
            hidSelfWeight[i] = ((Math.random() * (max - min)) + min) / I_dim;
            hidDelta[i] = 0.0;
        }

        outSelfWeight[0] = Math.random() / H_dim;
        outDelta[0] = 0.0;
    }

    // stadardises data between 0-1 range
    // Can also de standardise (In ARRAYS)
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
    
    // stadardises data between 0-1 range
    // Can also de standardise (In ARRAYS)

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

    // Calculates weighted sum of 2 arrays
    public static double wSum(double[] m1, double[] m2){
        double sum = 0.0;
        int size = m1.length;
        for (int i = 0; i < size; i++){
            sum += m1[i] * m2[i];
            
        }
        return sum;
    }

    // Calculates our error value
    public static double errorFunc(double predicted, double real){
        return Math.pow((predicted - real), 2);
    }

    // Activation function (Sigmoid)
    public static double activation(double x){
        //Sigmoid
        double result = 1 / (1 + Math.exp(-x));
        return result;
    }

    // Gets sum of an array
    public static double getOverall(double [] thisData){
        double sum = 0.0;
        int size = thisData.length;
        for (int i = 0; i < size; i++){
            sum += thisData[i];
        }
        return sum;
    }

    // Gets omega value for weight decay
    public static double getOmega(){
        int totalWeights = (H_dim * O_dim) + (H_dim * I_dim) + H_dim + O_dim;
        double weightSquared = 0.0;
        for (int i = 0; i < H_dim; i++){
            weightSquared += Math.pow(weightToOut[i], 2);
            for (int j = 0; j < I_dim; j++){
                weightSquared += Math.pow(weightToHid[i][j], 2);
            }
        }
        return weightSquared / (2 * totalWeights);
        
    }

    // Back prop algorithm
    public static void backProp(int dataIndex, double observed, double[][] thisData, int currentEpoch){
        // Values needed
        double desiredOutVal = normaliseSingle(observed, "pre");
        double[] inpVal = thisData[dataIndex];
        double upsilon = 1 / (learning_param * currentEpoch+1);
        double omega = getOmega();
        double upsilonOmega = upsilon * omega;
        
        // Calculates delta values and weight changes
        for (int i = 0; i < O_dim; i++){
            outDelta[i] = (desiredOutVal - outVal + upsilonOmega) * derivative(outVal);
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

    // Derivative of sigmoid
    private static double derivative(double x) {
        return (x * (1 - x));
    }

    //Feed forward algorithm

    public static double[] feedForward(int epochs, boolean training, double[][] useThisData, double[] useTheseOutputs, double breakCase) throws IOException{
        double[] epochErrors = new double[epochs];
        double[] results = new double[useThisData.length+1];
        double[][] trainingData = getData("training");
        for (int j = 0; j < epochs; j++){
            double[] errors = new double[useThisData.length];

            for (int i = 0; i < useThisData.length; i++){
                double[] thisPass = useThisData[i];
                for (int x = 0; x < H_dim; x++){
                    double[] weights = weightToHid[x];

                    double[] norm = normaliseData(thisPass, "pre"); 
                    double wS = wSum(weights, norm) + (hidSelfWeight[x] * hidVals[x]);

                    hidVals[x] = activation(wS);
                }

                for (int x = 0; x < O_dim; x++){
                    double wS = wSum(weightToOut, hidVals) + (outSelfWeight[x] * outVal);
                    outVal = activation(wS);

                    double error = errorFunc(outVal, normaliseSingle(useTheseOutputs[i], "pre"));
                    
                    //String results = String.format("Expected: %f             Got: %f", normaliseSingle(useTheseOutputs[i], "pre"), outVal);
                    //System.out.println(results);
                    errors[i] = error;
                }

                if (training){
                    backProp(i, useTheseOutputs[i], trainingData, j);
                } else {
                    results[i] = normaliseSingle(outVal, "post");
                    dotGot.add(normaliseSingle(outVal, "post"));
                    dotExpected.add(useTheseOutputs[i]);
                }

            }
            epochErrors[j] = Math.pow(getOverall(errors) / useThisData.length, 0.5);

            if (epochErrors[j] < breakCase && training){
                System.out.println(String.format("Successfuly Trained in %d Epochs", j));
                return(epochErrors);
            }
        } 

        if (training){
            System.out.println(String.format("Epoch Amount Hit Current Error: %f", epochErrors[epochErrors.length-1]));
            return(epochErrors);
        } else {
            return results;
        }
    }

    // Writes values to txt file for graph plotting
    public static void plotErrorGraph(ArrayList<Double> errorResults) throws IOException{
        new FileWriter("errorData.txt").close();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("errorData.txt"))){
            for (double line: errorResults){
                writer.write (line + "\n");
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace ();
        }
    }

    // Writes values to txt file for graph plotting
    public static void plotDot(ArrayList<Double> expected, ArrayList<Double> got) throws IOException{
        new FileWriter("dotGraph.txt").close();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("dotGraph.txt"))){
            for (int i = 0; i < expected.size(); i++){
                writer.write(String.format("%f,%f\n", expected.get(i), got.get(i)));
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace ();
        }
    }

    // Gets accuracy of the current model
    public static double getAccuracy(double[] thisTable, double[] desired) {
        double sum = 0.0;

        for (int i = 0; i < thisTable.length - 1; i++){
            double top = (Math.abs(desired[i] - thisTable[i]) / desired[i]) * 100;
            sum += top;
        }
        double percent = 100 - (sum / thisTable.length - 1);

        return percent;

    }

    // Creates user UI and runs code in correct Order
    public static void main(String[] args) throws IOException{

        /*
        double[][] theData = getData("validation");
        double[] theResults = getDesiredData("validation");

        for (int i = 0; i < theData.length; i++){
            System.out.println(String.format("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f", theData[i][0], theData[i][1], theData[i][2], theData[i][3], theData[i][4], theData[i][5], theData[i][6], theData[i][7], theResults[i]));
        }*/

        

        ArrayList<Double> errorResults = new ArrayList<Double>();

        System.out.println("Would you like to: TRAIN the AI, RUN the AI, predict the next value or STOP the program");
        Scanner userInp = new Scanner(System.in);
        String accInp = userInp.nextLine();

        while (!accInp.equalsIgnoreCase("stop")){

            if (accInp.equalsIgnoreCase("train")){
                initWeights();
                System.out.println("How Many Epochs");
                Scanner epochInp = new Scanner(System.in);
                int epoch = epochInp.nextInt();

                double[] tempVal = feedForward(epoch, true, getData("training"), getDesiredData("training"), 0.005);

                IntStream.iterate(0, i -> i + 1).limit(tempVal.length-1).forEach(i -> errorResults.add(tempVal[i]));

                System.out.println("Training Completed");
            }

            if (accInp.equalsIgnoreCase("run")){

                double[][] validation = getData("validation");
                double[] desired = getDesiredData("Validation");
            
                double[] thisTable = feedForward(1, false, validation, desired, 0.05);


                IntStream.iterate(1, i -> i + 1).limit(I_dim).forEach(i -> System.out.print(String.format("   Input%d    |", i)));
                System.out.print("   Expected    |");
                System.out.print("   Result    |");
                System.out.println();
                IntStream.iterate(0, i -> i + 1).limit(I_dim + 2).forEach(i -> System.out.print("--------------"));
                System.out.println("--");
                IntStream.iterate(0, i -> i + 1).limit(validation.length-1).forEach(i -> System.out.println(String.format("   %.1f   |   %.1f   |   %.1f   |   %.1f   |   %.1f   |   %.1f   |   %.1f   |   %.1f   |   %.6f   |   %.6f   |", data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], desired[i], thisTable[i])));
                //XOR IntStream.iterate(0, i -> i + 1).limit(data.length).forEach(i -> System.out.println(String.format("   %.1f   |   %.1f   |   %.6f   |   %.6f   |", data[i][0], data[i][1], desiredOut[i], thisTable[i])));
                
                double accuracy = getAccuracy(thisTable, desired);

                System.out.println(String.format("The accuracy of the model is %.2f%s", accuracy, "%"));

                System.out.println("Would you like to continue training until a new error threshold is met (1st run is 0.05). If so please enter a value or type no to cancel: ");
                Scanner errorVal = new Scanner(System.in);
                String accErrorVal = errorVal.nextLine();

                if (!accErrorVal.equalsIgnoreCase("no")){
                    System.out.println("Please enter epoch count");
                    Scanner epochInp = new Scanner(System.in);
                    int accEpochInp = Integer.parseInt(epochInp.nextLine());
                    double[] tempVal = feedForward(accEpochInp, true, getData("training"), getDesiredData("training"), Double.parseDouble(accErrorVal));
                    IntStream.iterate(0, i -> i + 1).limit(tempVal.length-1).forEach(i -> errorResults.add(tempVal[i]));
                }
            }

            if (accInp.equalsIgnoreCase("predict")){
                feedForward(1, false, getData("test"), getDesiredData("test"), 0.005);
                System.out.println(normaliseSingle(outVal, "post"));
            }

            System.out.println("Would you like to: TRAIN the AI, RUN the AI, predict the next value or STOP the program");
            userInp = new Scanner(System.in);
            accInp = userInp.nextLine().toLowerCase();
        }
        plotErrorGraph(errorResults);
        plotDot(dotExpected, dotGot);
    } 
}
