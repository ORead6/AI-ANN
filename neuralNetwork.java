public class neuralNetwork{

    public static int I_dim = 2;
    public static int H_dim = 2;
    public static int O_dim = 1;

    public static int epochCount = 10000;
    public static Double learning_param = 0.1;

    public static Double[][] weightToHid = new Double[I_dim][H_dim];
    public static Double[] weightToOut = new Double[H_dim];
    public static Double[] hidBias = new Double[H_dim];
    public static Double[] hidVals = new Double[H_dim];
    public static Double[] hidSelfWeight = new Double[H_dim]; 
    public static Double[] outSelfWeight = new Double[O_dim]; 
    public static Double outBias = Math.random() / H_dim;
    public static Double outVal = 0.0;
    public static Double[] outDelta = new Double[O_dim];
    public static Double[] hidDelta = new Double[H_dim];

    public static Double[][] data = getData();
    public static Double[] desiredOut = {0.0, 1.0, 1.0, 0.0};

    public static Double[][] getData(){
        Double[][] theData = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
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

    public static Double[] normaliseData(Double[] data, String thisCase){
        int minVal = 0;
        int maxVal = 1;

        if (thisCase == "pre"){
            Double[] answer = new Double[data.length];
            for (int i = 0; i < answer.length; i++){
                answer[i] = (data[i] - minVal) / (maxVal - minVal);
            }
            return answer;

        } else {
            Double[] answer = new Double[data.length];
            for (int i = 0; i < answer.length; i++){
                answer[i] = (data[i] * (maxVal - minVal)) + minVal;
            }
            return answer;
        }

        
    }

    public static Double normaliseSingle(Double data, String thisCase){
        int minVal = 0;
        int maxVal = 1;

        if (thisCase == "pre"){
            Double answer;
            answer = (data - minVal) / (maxVal - minVal);
            return answer;

        } else {
            Double answer;
            answer = (data * (maxVal - minVal)) + minVal;
            return answer;
        }
    }

    public static Double wSum(Double[] m1, Double[] m2){
        Double sum = 0.0;
        for (int i = 0; i < m1.length; i++){
            sum += m1[i] * m2[i];
            
        }
        return sum;
    }

    public static Double errorFunc(Double predicted, Double real){
        return Math.pow((predicted - real), 2);
    }

    public static Double activation(Double x){
        //Sigmoid
        Double result = 1 / (1 + Math.exp(-x));
        return result;
    }

    public static Double getOverall(Double[] thisData){
        Double sum = 0.0;
        for (int i = 0; i < thisData.length; i++){
            sum += thisData[i];
        }
        return sum;
    }

    public static void backProp(int dataIndex, Double observed){
        Double desiredOutVal = normaliseSingle(observed, "pre");
        Double[] inpVal = getData()[dataIndex];
        
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

    private static double derivative(Double x) {
        return (x * (1 - x));
    }

    public static Double[] feedForward(int epochs){
        Double[] epochErrors = new Double[epochs];
        for (int j = 0; j < epochs; j++){
            Double[] errors = new Double[getData().length];

            for (int i = 0; i < getData().length; i++){
                Double[] thisPass = getData()[i];
                for (int x = 0; x < H_dim; x++){
                    Double[] weights = weightToHid[x];

                    Double[] norm = normaliseData(thisPass, "pre"); 
                    Double wS = wSum(weights, norm) + (hidSelfWeight[x] * hidVals[x]);

                    hidVals[x] = activation(wS);
                }

                for (int x = 0; x < O_dim; x++){
                    Double wS = wSum(weightToOut, hidVals) + (outSelfWeight[x] * outVal);
                    outVal = activation(wS);

                    Double error = errorFunc(outVal, normaliseSingle(desiredOut[i], "pre"));
                    
                    String results = String.format("Expected: %f             Got: %f", desiredOut[i], outVal);
                    System.out.println(results);
                    errors[i] = error;
                }

                backProp(i, desiredOut[i]);

            }
            epochErrors[j] = getOverall(errors);
        } 

        return(epochErrors);
    }

    public static void plotGraph(Double[] errorGraph){
        
    }

    public static void main(String[] args){
        initWeights();
        plotGraph(feedForward(epochCount));

    }
}