import sys
import time
import signal
import argparse

from src.FFNEvaluation import sampleEval

# Register an handler for the timeout
def handler(signum, frame):
    raise Exception("")#kill running :: Timeout occurs")

def runSingleInstanceForAllCategory(onnxFile,vnnlibFile,timeout):
   'called from run_all_catergory.py'

   # Register the signal function handler
   signal.signal(signal.SIGALRM, handler)

   # Define a timeout for "runSingleInstance"
   signal.alarm(int(timeout))

   '"runSingleInstance" will continue until any adversarial found or timeout occurs'
   'When timeout occurs codes written within exception will be executed'
   try:
       retStatus = runSingleInstance(onnxFile,vnnlibFile)
       return retStatus
   except Exception as exc:
       #printStr = "timeout," + str(timeout) + "\n" 
       print(exc)

def runSingleInstance(onnxFile,vnnlibFile):
   #Variable Initialization
   startTime = time.time()

   onnxFileName = onnxFile.split('/')[-1]
   vnnFileName = vnnlibFile.split('/')[-1]

   print(f"\nTesting network model {onnxFileName} for property file {vnnFileName}")

   'Calling sampleEval until any adversarial found or timout ocuurs'
   while(1):
       status = sampleEval(onnxFile,vnnlibFile)
       endTime = time.time()
       timeElapsed = endTime - startTime
       #print("Time elapsed: ",timeElapsed)

       if (status == "violated"):
          resultStr = status+", "+str(round(timeElapsed,4))
          return resultStr
    
   #resultStr = "timeout,"+str(round(timeElapsed,4)) + "\n"
   #return resultStr


#Main function
if __name__ == '__main__':

   #Commandline arguments processing

   # Instantiate the parser
   parser = argparse.ArgumentParser(description='Optional app description')

   # Required onnx file path 
   parser.add_argument('-m',
                    help='A required onnx model file path')

   # Required vnnlib file path
   parser.add_argument('-p', 
                    help='A required vnnlib file path')

   # Optional resultfile path
   parser.add_argument('-o',
                    help='An optional result file path')

   # optional timeout parameter
   parser.add_argument('-t',
                    help='An optional timeout')

   args = parser.parse_args()
   onnxFile = args.m
   vnnlibFile = args.p
   
   'Check for the onnxfile in the commandline, it is a mandatory parameter'
   if (onnxFile is None):
      print ("\n!!! Failed to provide onnx file on the command line!")
      sys.exit(1)  # Exit from program

   'Check for the vnnlib file in the commandline, it is a mandatory parameter'
   if (vnnlibFile is None):
      print ("\n!!! Failed to provide vnnlib file path on the command line!")
      sys.exit(1)  # Exit from program


   resultFile = args.o 

   'Set default for resultFile if no result file is provided in the commandline'
   'It is an optional parameter'
   if ( resultFile is None ):
      resultFile = "out.txt"
      print ("\n!!! No result_file path is provided on the command line!")
      print("Output will be written in default result file- \"{0}\"".format(resultFile))
   else:
      print("\nOutput will be written in - \"{0}\"".format(resultFile))

   timeout = args.t

   'Set default for timeout if no timeout value is provided in the commandline'
   'It is an optional parameter'
   if ( timeout is None ):
      print ("\n!!! timeout is not on the command line!")
      print ("Default timeout is set as - 60 sec")
      timeout = 60.0
   else:
      print ("\ntimeout is  - {0} sec".format(timeout))


   # Register the signal function handler
   signal.signal(signal.SIGALRM, handler)

   # Define a timeout for "runSingleInstance"
   signal.alarm(int(timeout))
   
   '"runSingleInstance" will continue until any adversarial found or timeout occurs'
   'When timeout occurs codes written within exception will be executed'
   outFile = open(resultFile, "w")
   try:
       retStatus = runSingleInstance(onnxFile,vnnlibFile)
       print("\nOutput is written in - \"{0}\"".format(resultFile))
       
   except Exception as exc:
       print("\nOutput is written in - \"{0}\"".format(resultFile))
       retStatus = "timeout," + str(timeout) + "\n" 
       print(exc)

   outFile.write(retStatus)
   outFile.close()
