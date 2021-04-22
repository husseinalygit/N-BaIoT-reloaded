# N-BaIoT-reloaded
This is a code reproduction for the paper titled "N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders"

# Dependancies 
developed using python 3.7.9 
- numpy v 1.19.4
- pandas v 1.1.2
- sklearn v 0.24.1
- tensorflow v 1.15.0

# Usage guide

1- first download the dataset from the following link 
http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

2- Extract the dataset in the same folder as the script, where the dataset folder would have the following layout 
  dataset/
   - 1.benign.csv
   - 1.gafgyt.combo.csv
   - 1.gafgyt.junk.csv
   - .. etc
 
 3- run the preprocessing script that will generate a folder named processed containing a folder per device. Each folder would have 2 CSV files for testing and training. 
   - testing.csv: contains 1/3 of benign traffic, and all the malicious traffic 
   - training.csv: contains 2/3 of benign traffic 
 
 4- open the model script and choose the device Id to train on, and then run the script; the model weights would be saved in the models folder with the following naming "autoencdoer_{device_id}" 
 

# Refrence 
Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher, D., & Elovici, Y. (2018). N-baiot—network-based detection of iot botnet attacks using deep autoencoders. IEEE Pervasive Computing, 17(3), 12-22.
