FFN : Fast Falsification of Neural Networks using Property Directed Testing
----------------------------------------------------------------------------

A. Folder structure
   -------------------

   1. README
   2. src  -- contains python source files
   3. benchmarks -- contains different benchmark categories

      Each category mainly contains - 

      a) .onnx files -- NN is given in .onnx file format

      b) .vnnlib files  -- provide normalized input ranges and property specifications

      c) category_instance.csv files -- provide instances to be run for that category 

      Format of category_instance.csv is  - ".onnx file path,.vnnlib file path,timeout"

      -- These file paths provide path for .onnx files and .vnnlib files inside the category 

      -- To find the absolute path of onnx files and .vnnlib files for this category - 

         -- need to prepend the category folder path before the paths specified in this category_instance.csv
        

   4. run_single_instance.py -- script to run a single instancei, how to run is given in "C"
   5. run_all_categories.py --to run all instances from a given category, how to run us given in "C" 
   6. Dockerfile -contains docker build and run commands as discussed below
   7. requirements.txt -- contains dependency list those to be installed during docker build

      ---To create requirements.txt according to the dependencies of FFN project -
         
            pip3 install pipreqs

            pipreqs . --ignore benchmarks --force
       Note : benchmark directory contains some pthoon programs which are not needed to run FFN
  
   
B: Getting Started
   -------------------------
1. clone FFN repository 

         git clone https://github.com/DMoumita/FFN.git

2. Entering into FFN directory
      
         cd FFN

3-a. Run using Docker 

    #Intall Docker Engine - please refer https://docs.docker.com/engine/install/ubuntu/
    #The Dockerfile in FFN folder shows how to install all the dependencies (mostly python and numpy packages) and set up the environment. 

   To build an image
    
    sudo docker build . -t ffn_image 

   To get a shell after building the image:
  
    sudo docker run -i -t ffn_image bash
    
   Run a script without entering in to the the shell:
   
    sudo docker run -i -t ffn_image python3 <run_single_instance.py> <onnx_file> <vnnlibfile> [-- resultfile resultfile] [ -- timeout timeout]


3-b. Run without docker 


   tested on Ubuntu 18.04 and 20.04
   
     sudo apt update
     sudo apt install python3
     sudo apt install python3-pip
     pip3 install onnx==1.8.0
     pip3 install onnxruntime==1.8.0
     pip3 install numpy==1.17.4

     
C. Evaluation
   ---------------
1: To run a single instance
   ------------------------------
      python3 run_single_instance.py -m <onnx_file_path> -p <vnnlib_file_path> [-o result_file_path] [-t timeout_parameter]


Example run:

a.
   
      python3 run_single_instance.py -m benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx -p benchmarks/acasxu/prop_2.vnnlib -o report.txt -t 10
      
 ---It evaluates "acasxu" benchmark Property 2 for network ACASXU_run2a_1_1_batch_2000.nnet
 
 ---After evaluation, result is stored in "report.txt"
 
 ---timeout parameter is set as 10 sec

b. 
   
      python3 run_single_instance.py -m benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx -p benchmarks/acasxu/prop_2.vnnlib 

 ---It evaluates "acasxu" benchmark Property 2 for network ACASXU_run2a_1_1_batch_2000.nnet
 
 ---After evaluation, result is stored in default result file - "out.txt"
 
 ---timeout parameter is not mentioned, 60 sec is assigned as a default value for it

2: To run all instances of a given benchmark category (from "benchmark" folder)
   ---------------------------------------------------------------------------
      python3 run_all_categories.py  [-c category] [-o result_file_path]

Note: runs all the instances from category/caegory_instance.csv file

Example run:

a. 

      python3 run_all_categories.py -c acasxu -o report.txt 

 ---It evaluates all the instances (.onnx files and .vnnlib files) from "acasxu/acasxu_instance.csv" files
 
 ---timeout value for each of the instance is mentioned in this .csv file
 
 ---After evaluation result is stored in report.txt
 

b.

      python3 run_all_categories.py 

 ---If no category is mentioned, "test" is considered as a default category
 ---It evaluates all networks(.onnx files in test directory) for all the properies(all .vnnlib files in test directory) from "test" benchmark category 
 
 ---If result_file_path is not mentioned, "report_test.txt"  is considered as a default result_file_path to store the result after evaluation

***Note: Since FFN has randomization, results may vary accross the runs.
