************************
*********THESIS*********
************************

This is the program for the thesis entitled "Facial Recognition System by Using a Three Triangle Method Utilizing OpenCV and Dlib"

The basic approach for this Facial Recognition is it uses only 4 points of the Face namely: 
the Left and Right Lateral Canthus, the Nasal Bridge and the Apex of the Nose. 
Creating 3 triangles around the face. Using basic trigonometry, the recognition of faces will then commence.

# Process 

## Training

### Dataset Preparation
Input Image -> Calculate the angle between two points using arctan -> Scaling image using Euclidian Distance Formula -> Save Image as Dataset

### Dataset Conversion
Fetch the Images -> Convert to Array -> Save as npy

### Training the Model
Load the npy -> Reshape the feature npy(x) (1, arrayshape[0]*arrayshape[1]) -> Split the data 0.30 -> Normalize the data -> Fit the data model.fit(x_train, y_train) -> Save the Model

## Recognition
Load Haar Cascade, Load Classifier, Load Shape Predictor -> Get the camera feed -> Convert image to Grayscale -> Detect faces -> Points to Rect -> Pass to Predictor (grayscale image, Rect) -> Get the shapes -> Align Image for slant faces -> Predict Faces

# Legends/Boilerplate

##### Application.py = GUI containing the Recognition.py methods
##### Alignment.py = Aligns the image using angle between two points and scales images using Euclidian Distance
##### bbrect.py = Converts the point to Dlib Rect to be used by Dlib
##### dataset_generator.py = This generates dataset from image feed I.e.: video
##### dataset_preprocess.py = This preprocess the files on 'unprocessed' folder to be used as dataset
##### image_test.py = This tests the images(on certain directory) and generates Confusion Matrix and Classification Report
##### landmark_generate.py = This parses the XML file to be used on our model generation (4 Point Landmark)
##### model_generate.py = This contains the options for model generation and this is the responsible for the creation of dlib model(4 Point Landmark)
##### Recognition.py = This is the output of the program. This contains the recognition part
##### training.py = This is the responsible for our SVM model.

# Files
##### custom_landmark_ver3 = Bugged model the nose point isn't on Apex
##### custom_landmark_ver4 = Final Model generated from Dlib's Training
##### svm_classifier_poly = SVM Working model for 5 Class (Random Lady, Me, Robert Downey JR., Elizabeth Olsen, Chris Evans)
##### report.txt/report_poly.txt = Classification Report

# Parameters
params.compute() in parameters_computation.py, this returns a results list containing the computed parameters.
The parameter mappings are as follows:

    Lines in the triangles, defined by two points
    l_ab = results[0][0]
    l_ac = results[0][1]
    l_ad = results[0][2]
    l_bc = results[0][3]
    l_bd = results[0][4]
    l_cd = results[0][5]

    Angles in Inner Triangle ACD
    a_cad = results[1][0]
    a_acd = results[1][1]
    a_adc = results[1][2]

    Angles in Inner Triangle BCD
    a_bcd = results[2][0]
    a_cbd = results[2][1]
    a_bdc = results[2][2]

    Angles in Outer Triangle ABD
    a_bad = results[3][0]
    a_abd = results[3][1]
    a_adb = results[3][2]

    Area of the Inner Triangles
    ar_acd = results[4][0]
    ar_bcd = results[4][1]

    Area of Outer Triangle
    ar_abd = results[5]
    
# Conversion of px to mm
This is in convert_to_mm() in parameters_computation.py
This is obtained by saving the extracted raw line measurements (X), getting their mean
And matching it with the corresponding actual line measurements of the face (Y)
With this 6 points, curve fitting was used to get the equation in the method
For a more accurate curve, it is recommended to generate more points for fitting.
See px_to_mm.xls for the values used

# Parameter Output
Change the output in the GUI by altering the prediction_text string in the run_recognition() in Application.py

    prediction_text = "Prediction: " + str(np.argmax(prediction))

Changes to this string will change the prediction output in both main window and output window

# Summary
Originally the proposed method was to compute the lines: AB • AC • AD • BC • BD • CD 

The angles: CAD • CBD • BCD • ACD • ADC • BDC

And the areas: Triangle ABD • Triangle ADC • Triangle BCD
