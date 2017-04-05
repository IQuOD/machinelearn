README file for CSIRO Project - Exploring machine learning to quality control ocean historic XBT temperature data. 

By Austin X Shen
________________________________________________________________________________________

Inside this package there are a mixture of scripts used to run machine learning algorithms, created packages that the scripts are dependent on, neural network training data, filename data and some directories storing selected tests. All files used over the course of the program are contained in this directory and so some older tests and programs that are not relevant to the final program will exist. I will categorise each of the files below and give a brief description about what the script/code/data does. The listing of the files will be in alphabetical order (or as they have appeared in on my screen). At the end of this I will discuss the key scripts and libraries that are required for execution of the final bottom flags program.

________________________________________________________________________________________

Scripts

1. bottom_flags_combine.py
This code is the final script that takes a the text file containing the netCDF data, the directory of the netCDF data, the neural network feature output file name and the results file, and will return the results of the neural network to the screen as well as generate the files that were successful. Currently, the code is written to take a default file name in, but this can be commented out to allow for a user input.

2. bottom_flags_points.py
This is a script that identifies the exact points of the hit bottom from the features extracted from the potential hit bottom and bad data points. It takes the same inputs as the bottom_flags_combine.py script and gives the statistical results for the hit bottom point detection. This takes the raw data, identifies the features and then runs the network all in one step (this was done in separate scripts previously).

3. bottom_flags_region.py
This attempts to identify the regions of a profile that are anomalous and highly likely to contain the hit bottom point. Like the two scripts mentioned above, this is the full package for the bottom flags region identification where all of the feature extraction and training of the network are done in one script. The output is a set of regions (can be and is likely to be more than one region) that are likely to contain the hit bottom point based on the network.

4. bottom_flags_WB.py
This identifies the wire breaks in a profile. Same as above, this script does all the work to identify the wire breaks in a user-inputted file set. 

5. compute.py
This is to optimise the performance of the functions in “hitbottom.py” (package) that detect important points in a profile. These functions are those to find the 	number of bad data points (constant temperature/increasing temperature) and potential hit bottom points. This optimises the performance by testing these functions over a selected parameter range and writes the number of points detected in a file (filename specified in the script).

6. nn1.py
The first neural network running script. This uses the TFlearn library to construct the network architecture, to train the model and to test the model against scientific QC flags. Note that this network is attempting to identify the exact location of the hit bottom points from the features provided.

7. nn2.py
The second neural network to identify the location of the hit bottom precisely. Takes in the updated features.

8. plot_stats.py
This takes in a text file of results from optimising functions and is used to generate useful figures to show the optimised parameters for the most effective bad data and potential hit bottom point detections.

9. profiles.py
A script that takes in a list of filenames (that point to the relevant .nc files) and plots the profiles. It also shows the line of bathymetry, the gradient profile and the scientific QC point.

10. reg_anomaly_detect.py
Attempts to use the anomaly detection unsupervised learning algorithm to find the regions of a profile that are more likely than not. This test was not as effective as other tests and so is not used further.

11. reg_identify.py
Identifies regions in a profile that are likely to contain the hit bottom. This uses a neural network rather than anomaly detection as the machine learning algorithm. 

12. reg_prep_network.py
This takes in the data from the regional analysis of the profiles and converts this information into features to use for the neural network in reg_identify.py. The features are written into a text file, which is read by the reg_identify script.

13. wb_identify.py
This is a script to identify the wire breaks in a profile (if there are any). This is a very effective decision tree test that does not use any machine learning.

14. zzz_bottom_flags.py
Repeated program - kept as a save file in case mistakes were made on the primary file.

________________________________________________________________________________________

Packages/Libraries

1. hitbottom.py
Essential library used to extract key points in a profile from the raw XBT data. This reads the data as .nc format and then returns lists of the key points in a profile (potential hit bottom points, bad data points and bathymetry data).

2. neuralnet.py
Was an attempt to write a neural network from scratch but found using pre-written libraries to be far more effective. After this was used to manipulate the input data in future neural networks.

3. nn_prepare_network.py
A set of functions to assist the user in preparing the input features and cross-validation features for a neural network. This includes the functions to remove repeats and extract features from the potential hit bottom points and bad data points in a profile. 

4. prepare_network.py
A script containing functions to produce the new features as well as the original features (after the first neural network was written and tested). This is the package that is often called in other scripts since it is most up to date. 

5. region_lib.py
Library of all key functions required for generating features for the hit bottom region analysis tests and running the machine learning algorithms (neural network and anomaly detection).

6. tf_neuralnet.py
Another script used to prepare the features of a neural network. This one appears to work on the first hit bottom point detection neural network. 

7. wb_library.py
Library of function used to detect wire breaks in a given profile.

________________________________________________________________________________________

Data

1. crossvalidation.txt
This file contains the names of the XBT drops with known hit bottoms from scientific QC (each of which has a list of temperature and depth data to create a profile) that are used in the cross-validation stage of the neural network. The original environment contained (in a directory above this one) a file containing the netCDF data, with corresponding filenames to those listed here. 

2. final_test_files.txt
Contains the names of the netCDF files for the XBT drops used for the final test (on random XBT data from a line). 

3. final_test_input.txt
Contains the names of the key files required to run the bottom_flags_combine.py script. This was simply to make inputting the user required files simpler.

4. HB_content.txt
All of the file names contained in a file (named HB_content) in a directory above this one. This was all of the scientific QC hit bottom profiles identified in a region of moderate latitude (around Australia). 

5. hb_wpothb_success.txt
Results for potential hit bottom test detection (identifying the success rate for various parameters).

6. HBfiles_golden.txt
Set of hit bottom files (.nc) that contain high quality scientific quality control (HB locations flagged were precise when checked by Rebecca Cowley).

7. initial_test_files.txt
A subset of the hit bottom data file names used for a test (I forgot which test this was for, but it is unlikely that this is an essential file since it was used at the start of the program). 

8. nn_complete_training.txt
A set of features used for training (first iteration of the hit bottom point identification) from the entire available training set of data. 

9. nn_crossvalidation_data1.txt
Neural network input features for the cross-validation step of identifying the hit bottom location precisely. These are the first iteration of features. Depth and temperature included.

10. nn_crossvalidation_data2.txt
Cross-validation features for identifying hit bottom points in a profile. These are the revised features (more identified in an attempt to increase the success rate).

11. nn_crossvalidation_data.txt
First cross-validation features identified for training neural network to find hit bottom points in a profile. 

12. nn_filecheck.txt
A selection of the files to check (visually inspect) from the first neural network. It is likely to be the profiles that were unsuccessfully detected by the network. Whether these profiles were the success or failure cases would be written in nn1.py or nn2.py.

13. nn_filecorrect.txt
The correctly detected files from the neural network (nn1.py or nn2.py) listed so that they could be visually inspected.

14. nn_golden_training.txt
Training data set (with features extracted) to be fed into the first neural network from the “golden” data set - that is the data that has been identified as precisely quality controlled for hit bottoms by scientists. 

15. reg_crossvalidation.txt
A list of features used for the cross-validation phase of the neural network training for the region detection algorithm. 

16. reg_test.txt
Test features for the hit bottom region test. 

17. reg_training.txt
Training features for the hit bottom region test.

18. stats_const_temp.txt
Optimisation results from the constant temperature point identification function by varying key parameters for the function. These points contribute to the bad data points in a profile.

19. stats_grad_spike.txt
Optimisation results from the gradient spike point identification test by varying parameters of the function. The points contribute to the potential hit bottom points in a profile.

20. stats_T_spike.txt
Optimisation results from the temperature spike point identification test by varying parameters for the function. The points contribute to the potential hit bottom points in a profile.

21. stats_temp_increase.txt
Optimisation results from the temperature increase identification test by varying parameters for the function. The points contribute to the bad data points in a profile.

22. subsetHB.txt
A subset of all scientific QC hit bottom profiles. Filenames referring to data in a separate file in a directory above this one.

23. t1points_results.txt
Results for the first test run of the final combined hit bottom region, point and wire break test (which was applied to a combination of known hit bottom and wire break tests. This was a run (of the bottom_flags_combine.py script) just before an unknown set was used and tested.

24. t1_points_test1.txt
The input features for the point identification neural network (first test run of final bottom flags test). 

25. t1WB_results.txt
Results of the wire break test script (all of the files were checked for wire breaks).

26. test1_input.txt
Just a text file containing the names of the files which are inputs for the bottom_flags_combine.py script (made to simplify running the script repeatedly - rather than taking a user input it would use the file names quoted in this text file). 

27. training.txt
Names of XBT data files that are used as training information for the neural network (features for training are extracted from these known HB profiles).

28. WB_content.txt
Names of known wire break profiles used to identify methods of detecting wire breaks. This should correspond to the netCDF files in a sub directory also titled “WB_content”.

29. wb_optimise.txt
Results of the wire break test for various parameters (some detection threshold) that determine the detection rate. 

30. WBfiles_golden.txt
Names of the netCDF files with known, precise wire break scientific quality control. 

________________________________________________________________________________________

Directories

1. __pycache__
This was automatically created when I called other python scripts as libraries in other scripts. I think it gets created even when other scripts are run anyway.

2. fin1
This file contains the outputted results for the final set of data that the bottom_flags_combine.py script is tested on. This contains the features extracted from the data available for the final test data set and the results obtained.

3. final_test
Contains the netCDF data of the profiles used for a final test of the completed bottom flags program. These files contain a mixture of hit bottoms, wire breaks and completely good data, and so are a good test of the true success rate of the program developed. I felt this data was worth keeping in the directory. All of the data provided has expert QC flags in the metadata. 

4. test1
This contains two text files with the extracted features for a neural network input for the first test of the bottom flags program. One of the files is for the features of the precise hit bottom location neural network, while the second is for identifying the region that is likely to contain the hit bottom point. 

________________________________________________________________________________________

Key files for final program

Libraries:

- hitbottom.py
- neuralnet.py
- tf_neuralnet.py
- prepare_network.py
- wb_library.py
- region_lib.py


Relevant data files:

- final_test_input.txt

This file should contain the names of the file where the netCDF files (actual profile data) is contained, the name of the text files where these profiles are listed, the desired name of the output feature file (when the features are extracted), the name of the neural network result file, and a prefix for the output files. A user can manually add these in if they choose (it will automatically revert to this if “final_test_input.txt” does not exist in the directory) but it is faster and easier to write a file titled “final_test_input.txt” and list the relevant files separated by commas. If a template is needed, this file currently exists in the directory. The name can be changed in the script (line 183). Needless to say, the files required or called by this “final_test_input.txt” are also required. 

To summarise, the files required are:
1. name of directory where netCDF files (profile data) is stored
2. filename of txt file containing all netCDF files

________________________________________________________________________________________