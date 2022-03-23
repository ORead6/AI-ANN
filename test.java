import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

public class test {

    public static void getData(double[][] inputs, double[] outputs){
        //XOR
        //double[][] theData = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        try {
            ArrayList<String[]> theData1 = new ArrayList<String[]>();
            ArrayList<String> desiredOut1 = new ArrayList<String>();
            BufferedReader csvReader = new BufferedReader(new FileReader("DataSet.csv"));
            String row;
            int x = 0;
            while ((row = csvReader.readLine()) != null) {
                if (x > 1){
                    String[] data = row.split(",");
                    theData1.add(data);
                    desiredOut1.add(data[3]);
                } else {x++;}
            }

            csvReader.close();

            System.out.println(theData1.size());

            double[][] theData = new double[8][theData1.size()];
            double[] desiredOut = new double[desiredOut1.size()];

            int size = desiredOut1.size();
            for (int i = 0; i < size; i++){
                desiredOut[i] = Double.parseDouble(desiredOut1.get(i));
            }

            int size2 = theData1.size();
            for (int i = 0; i < size2; i++){
                int size3 = theData1.get(i).length;
                double[] temp = new double[size3];
                for (int y = 0; y < size3; y++){
                    temp[y] = Double.parseDouble(theData1.get(i)[y]);
                }

                theData[i] = temp;
            }

             inputs = theData;
             outputs = desiredOut;


        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
    
    public static void main(String[] args){
        getData();
    }
}
