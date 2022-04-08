import sys
import csv
import signal
import argparse
from run_single_instance import runSingleInstanceForAllCategory


'Main function'

#Commandline arguments processing

# Instantiate the parser
parser = argparse.ArgumentParser(description='FFN run all instances description')

# Optional benchmark category
parser.add_argument('-c',
                    help='An optional benchmark category')

# Optional result filepath
parser.add_argument('-o',
                    help='An optional result file path')

args = parser.parse_args()
category = args.c

'Set default category if no category is specified in commandline'
'It is an optional parameter'
if ( category is None ):
    category = "test"
    print ("\n!!! No benchmark category is provided on the command line!")
    print ("Default benchmark category is taken - \"test\"")
else:
    print ("Testing benchmark category - \"{0}\"".format(category))

reportFile = args.o

'Set default resultFile path if no result file is given in commandline'
'It is an optional parameter'
if ( reportFile is None ):
    reportFile ="report_"+category+".txt"
    print ("\n!!! No result_file is provided on the command line!")
    print ("Taking default result_file -\"{0}\"".format(reportFile))


#Reading cat_instance.csv for .onnx file path, .vnnlib file path and timeout

catInsCsvFile = "benchmarks/"+category+"/"+category+"_instances.csv"
insCsvFile = open(catInsCsvFile, 'r')
outFile = open(reportFile, 'w')
reader = csv.reader(insCsvFile)

for row in reader:
    onnxFile = "benchmarks/"+category+"/"+row[0]
    if (onnxFile.endswith('.onnx') == False):
       print("\n!!!Wrong onnx file format for -\"{0}\"".format(onnxFile))
       continue
    vnnlibFile = "benchmarks/"+category+"/"+row[1]
    if (vnnlibFile.endswith('.vnnlib') == False):
       print("\n!!!Wrong vnnlib file format for -\"{0}\"".format(vnnlibFile))
       continue
    timeout = row[2]

    resultStr = runSingleInstanceForAllCategory(onnxFile,vnnlibFile,timeout)
    if (not resultStr):
       resultStr = "timeout,"+timeout

    printStr=onnxFile+","+vnnlibFile+","+resultStr+"\n"
    outFile.write(printStr)

insCsvFile.close()
outFile.close()

