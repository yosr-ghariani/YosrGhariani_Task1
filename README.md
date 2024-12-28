

# *Blink Detection Python Script* 
This computer vision project detects and calculates eye blinks using facial landmarks of the MediaPipe library.

---

## **Objective**  
The objective of this project is to develop a Python program that detects eye blinks in real-time using a webcam. The program processes the video stream and displays the following information:  

1. A real-time counter for the number of blinks detected.  
2. The average Eye Aspect Ratio (EAR) for monitoring eye behavior.

---

## **Research**

### **1. Existing Solutions**  
Here are the sources that inspired the workflow:  

- [A github repository using the EAR equation and the dlib model](https://github.com/nourhenehanana/Eye-Blink-Detection)
- [A research paper about The Assistance of Eye Blink Detection for Two- Factor Authentication](https://www.researchgate.net/publication/373895993_The_Assistance_of_Eye_Blink_Detection_for_Two-_Factor_Authentication)
- [A github repository using Mediapipe and OpenCV and the euclidien distance](https://github.com/Shakirsadiq6/Blink_Detection_Python)

  
### **2. Exploring Models and Libraries**
- **[shape_predictor_68_face_landmarks](https://www.restack.io/p/open-source-face-recognition-system-answer-dlib-shape-predictor-68-download-cat-ai):**  A face-landmark model from the dlib library that uses 68 landmarks to identify main facial features. While this model is popular, it posed challenges for setup and use, particularly on Google Colab due to dependency conflicts.  

- **[MediaPipe Face Landmarker - FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker):**  MediaPipe offers an efficient and lightweight solution for facial landmark detection, specifically the FaceMesh model, which we used for extracting eye landmarks. This model provided smoother integration with OpenCV.

---

### **3. Webcam Detection**  
Initially, we explored webcam detection using Google Colab, following a [tutorial](https://www.youtube.com/watch?v=YjWh7QvVH60) that integrated JSON and the Haar Cascade Classifier. While functional, it was challenging to adapt this approach for landmark-based EAR calculations. Consequently, we shifted to local webcam detection using OpenCV for better control and simplicity.  

---

### **4. EAR (Eye Aspect Ratio)**  
The **[Eye Aspect Ratio (EAR)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044337/)** is a mathematical formula used to estimate the openness of the eye. By tracking specific vertical and horizontal distances between the eye's landmarks, the EAR helps us determine if the eyes are open or closed.  

#### EAR Equation: 
![Screenshot 2024-12-28 162817](https://github.com/user-attachments/assets/57875354-0b99-4653-a7b4-58c57ba9d7db)
 

---

## **Implementation Steps**  
- **Selecting Landmarks:**  
   The most important eye landmarks were chosen from the MediaPipe Face Landmarker.
  ![Mediapipe eyes landmarks](https://github.com/user-attachments/assets/647b9a59-15d0-44c7-8b6b-51a360041f50)
- **Calculating EAR:**  
   A function was implemented to calculate the EAR using the selected landmarks.  
- **Video Stream and Detection:**  
   Using OpenCV, the program processes the webcam feed, calculates the EAR in real-time, and counts blinks based on a predefined threshold and consecutive frame count.  
- **Displaying Results:**  
   The real-time video feed includes the EAR, blink counter, and facial landmarks for visual feedback.  
- **Final Summary:**  
   After the set duration, the program prints the total number of blinks detected.

---

## **Setup**  
To set up the code, install the following dependencies (mainly OpenCV, NumPy, and MediaPipe).  

Please note that [MediaPipe Python PyPI](https://mediapipe.readthedocs.io/en/latest/getting_started/troubleshooting.html#:~:text=after%20running%20pip%20install%20mediapipe,x86_64%20macOS%2010.15%2B) officially supports Python versions 3.7 to 3.10 on 64-bit systems.  

Run the following command to install all required packages:  
```bash
pip install -r requirements.txt
```

Run the following command to run the script:
```bash
python Blink_Detection.py
```

## Outcome

Below is an example output to demonstrate the eye blink detection and the ever-changing EAR value: 
![example screenshot](https://github.com/user-attachments/assets/fcad2456-4f8b-4dfd-b280-fa12a656016d )

## Perspectives

To conclude, this Python program effectively detects and counts eye blinks in real-time using the Eye Aspect Ratio (EAR) and facial landmarks provided by MediaPipe. However, there is room for improvement:  

- The EAR threshold can sometimes lead to false positives, as it is set somewhat high. A more adaptive thresholding mechanism step could enhance accuracy.  
- Using additional or more detailed landmarks around the eyes might help refine the EAR calculation and better capture subtle differences in eye closure.  
- Improving the robustness of blink detection, especially in challenging conditions like different head angles.  



