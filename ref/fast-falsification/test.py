from run_single_instance import runSingleInstanceForAllCategory
import csv


if __name__ == '__main__':
    # category = 'acasxu'
    # catInsCsvFile = "benchmarks/"+category+"/"+category+"_instances.csv"
    # insCsvFile = open(catInsCsvFile, 'r')
    # reader = csv.reader(insCsvFile)
    # for row in reader:
    #     onnxFile = "benchmarks/"+category+"/"+row[0]
    #     vnnlibFile = "benchmarks/"+category+"/"+row[1]
    #     timeout = row[2]
    #     print(onnxFile, vnnlibFile)
    #     retStatus = runSingleInstance(onnxFile,vnnlibFile)
    #     print(retStatus)
    #     break

    onnxFile = 'benchmarks/acasxu/ACASXU_run2a_1_7_batch_2000.onnx'
    vnnlibFile = 'benchmarks/acasxu/prop_3.vnnlib'
    retStatus = runSingleInstanceForAllCategory(onnxFile, vnnlibFile, 2)
    print(retStatus)