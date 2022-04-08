import numpy as np
import re
from itertools import chain

import onnx
import onnxruntime as ort

import numpy as np

def predictWithOnnxruntime(modelDef, *inputs):
    'run an onnx model'
    
    sess = ort.InferenceSession(modelDef.SerializeToString())
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    return res[0]

def removeUnusedInitializers(model):
    'return a modified model'

    newInit = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            newInit.append(init)
        else:
            print(f"removing unused initializer {init.name}")

    graph = onnx.helper.make_graph(model.graph.node, model.graph.name, model.graph.input,
                                   model.graph.output, newInit)

    onnxModel = makeModelWithGraph(model, graph)

    return onnxModel

def makeModelWithGraph(model, graph, ir_version=None, checkModel=True):
    'copy a model with a new graph'

    onnxModel = onnx.helper.make_model(graph)
    onnxModel.ir_version = ir_version if ir_version is not None else model.ir_version
    onnxModel.producer_name = model.producer_name
    onnxModel.producer_version = model.producer_version
    onnxModel.domain = model.domain
    onnxModel.model_version = model.model_version
    onnxModel.doc_string = model.doc_string

    #print(f"making model with ir version: {model.irVersion}")
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnxModel, values)

    # fix opset import
    for oimp in model.opset_import:
        opSet = onnxModel.opset_import.add()
        opSet.domain = oimp.domain
        opSet.version = oimp.version

    if checkModel:
        onnx.checker.check_model(onnxModel, full_check=True)

    return onnxModel

def findObjectiveFuncionType(spec,numOutputs):
   'find target output label ant its objective type from property spec matrix'

   #initilization with 0
   targetDic=dict.fromkeys(range(numOutputs),0)
   objDic=dict.fromkeys(range(numOutputs),0)

   '''Check all binary expression to find output variable with 
   maximum nonzero value (target label) and maximum negative value
   if both are same then target label is for maximization since 
   <= used in all expressions, else target label is for minimization
   '''

   for j in range(len(spec)):
       arr=spec[j][0] #get mat from propery matrix -[mat,rhs]
       for k in range(len(arr)):
           for kk in range(len(arr[k])):
              if (arr[k][kk] != 0):
                 cntr=targetDic.get(kk)+1
                 targetDic[kk]=cntr
              if (arr[k][kk] < 0):
                 cntr=objDic.get(kk)+1
                 objDic[kk]=cntr

   target=max(targetDic, key=targetDic.get)
   objType=max(objDic, key=objDic.get)

   output=[]
   output.append(target)

   if (target == objType) :
      output.append(0) #Maximization
      #print("Target:",target,"Objective :Max",)
   else:
      output.append(1) #Minimization
      #print("Target:",target,"Objective :Min",)
   return output

def checkAndSegregateSamplesForMaximization(posSample,negSample, smple,oldPos,targetNode):      
    a=smple[0][1]
    large = a[targetNode]
    lastIndex=0
    posSample.append(smple[0])
    for i in range(1,len(smple)):
        a=smple[i][1]
        newLarge = a[targetNode]
        if newLarge > large :
            posSample.remove(smple[lastIndex])
            posSample.append(smple[i])
            negSample.append(smple[lastIndex])
            lastIndex=i
            large=newLarge
        else:
            if newLarge < large :
               negSample.append(smple[i])
    if (len(oldPos) > 0) :
        cPos=posSample[0][1]
        oldPos1=oldPos[0][1]
        if ( oldPos1[targetNode] > cPos[targetNode] ) :
            negSample.append(posSample[0])
            posSample.remove(posSample[0])
            posSample.append(oldPos[0])

def checkAndSegregateSamplesForMinimization(posSample,negSample, smple,oldPos,targetNode):      
    a=smple[0][1]
    small = a[targetNode]
    lastIndex=0
    posSample.append(smple[0])
    for i in range(1,len(smple)):
        a=smple[i][1]
        newSmall = a[targetNode]
        if newSmall < small :
            posSample.remove(smple[lastIndex])
            posSample.append(smple[i])
            negSample.append(smple[lastIndex])
            lastIndex=i
            small=newSmall
        else:
            if newSmall < small :
               negSample.append(smple[i])
    if (len(oldPos) > 0) :
        cPos=posSample[0][1]
        oldPos1=oldPos[0][1]
        if ( oldPos1[targetNode] < cPos[targetNode] ) :
            negSample.append(posSample[0])
            posSample.remove(posSample[0])
            posSample.append(oldPos[0])



