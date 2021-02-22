/*
 * Stolen from http://xml.nig.ac.jp/tutorial/rest/index.html
 * and http://www.dr-chuck.com/csev-blog/2007/09/calling-rest-web-services-from-java/
*/
import java.io.*;
import java.net.*;

public class Rest {

    public static void main(String[] args) throws IOException {
        URL url = new URL(INSERT_HERE_YOUR_URL);
        String query = INSERT_HERE_YOUR_URL_PARAMETERS;

        //make connection
        URLConnection urlc = url.openConnection();

        //use post mode
        urlc.setDoOutput(true);
        urlc.setAllowUserInteraction(false);

        //send query
        PrintStream ps = new PrintStream(urlc.getOutputStream());
        ps.print(query);
        ps.close();

        //get result
        BufferedReader br = new BufferedReader(new InputStreamReader(urlc
            .getInputStream()));
        String l = null;
        while ((l=br.readLine())!=null) {
            System.out.println(l);
        }
        br.close();
    }