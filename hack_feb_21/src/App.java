
import java.awt.Graphics2D;
import java.awt.image.BufferStrategy;
import java.awt.*;
import javax.swing.JFrame;
import javax.swing.JPanel;

//Image Rotation
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;

//Keyboard and Mouse
import java.awt.event.*;
import java.util.Arrays;

public class App {

    //Declare the variables needed for the graphics
    public JFrame frame;
    public Canvas canvas;
    public BufferStrategy bufferStrategy;

    public static void main(String[] args) throws Exception {
        System.out.println("Hello, World!");
    }
    public void init() {

        Image[] clu = new Image[13];
        for (int i = 0; i < clubs.length; i++ ) {
             clubs[i] = getImage(getDocumentBase(), "c" + (i + 1) + ".gif");
         }
     }
}
