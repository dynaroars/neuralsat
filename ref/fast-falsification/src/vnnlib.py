#vnnlib simple utilities

from copy import deepcopy
import re

import numpy as np

import onnxruntime as ort
import onnx

def readStatements(vnnlibFilename):
    '''process vnnlib and return a list of strings (statements)

    useful to get rid of comments and blank lines and combine multi-line statements
    '''

    with open(vnnlibFilename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    assert len(lines) > 0

    # combine lines if case a single command spans multiple lines
    openParentheses = 0
    statements = []
    currentStatement = ''
    
    for line in lines:
        commentIndex = line.find(';')

        if commentIndex != -1:
            line = line[:commentIndex].rstrip()
        
        if not line:
            continue

        newOpen = line.count('(')
        newClose = line.count(')')

        openParentheses += newOpen - newClose

        assert openParentheses >= 0, "mismatched parenthesis in vnnlib file"

        # add space
        currentStatement += ' ' if currentStatement else ''
        currentStatement += line

        if openParentheses == 0:
            statements.append(currentStatement)
            currentStatement = ''

    if currentStatement:
        statements.append(currentStatement)

    # remove repeated whitespace characters
    statements = [" ".join(s.split()) for s in statements]

    # remove space after '('
    statements = [s.replace('( ', '(') for s in statements]

    # remove space after ')'
    statements = [s.replace(') ', ')') for s in statements]

    return statements

def updateRvTuple(rvTuple, op, first, second, numInputs, numOutputs):
    'update tuple from rv in readVnnlib, with the passed in constraint "(op first second)"'
    
    if first.startswith("X_"):
        # Input constraints
        index = int(first[2:])

        assert not second.startswith("X") and not second.startswith("Y"), \
                                     f"input constraints must be box ({op} {first} {second})"
        assert 0 <= index < numInputs

        limits = rvTuple[0][index]
        
        if op == "<=":
            limits[1] = min(float(second), limits[1])
        else:
            limits[0] = max(float(second), limits[0])

        assert limits[0] <= limits[1], f"{first} range is empty: {limits}"

    else:
        # output constraint
        if op == ">=":
            # swap order if op is >=
            first, second = second, first

        row = [0.0] * numOutputs
        rhs = 0.0

        # assume op is <=
        if first.startswith("Y_") and second.startswith("Y_"):
            index1 = int(first[2:])
            index2 = int(second[2:])

            row[index1] = 1
            row[index2] = -1
        elif first.startswith("Y_"):
            index1 = int(first[2:])
            row[index1] = 1
            rhs = float(second)
        else:
            assert second.startswith("Y_")
            index2 = int(second[2:])
            row[index2] = -1
            rhs = -1 * float(first)

        mat, rhsList = rvTuple[1], rvTuple[2]
        mat.append(row)
        rhsList.append(rhs)

def makeInputBoxDict(numInputs):
    'make a dict for the input box'

    rv = {i: [-np.inf, np.inf] for i in range(numInputs)}

    return rv

def getIoNodes(onnxModel):
    'returns 3 -tuple: input node, output nodes, input dtype'

    sess = ort.InferenceSession(onnxModel.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    inputName = inputs[0]

    outputs = [o.name for o in sess.get_outputs()]
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"
    outputName = outputs[0]

    g = onnxModel.graph
    inp = [n for n in g.input if n.name == inputName][0]
    out = [n for n in g.output if n.name == outputName][0]

    inputType = g.input[0].type.tensor_type.elem_type

    assert inputType in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    dtype = np.float32 if inputType == onnx.TensorProto.FLOAT else np.float64

    return inp, out, dtype

def get_numInputs_outputs(onnx_filename):
    'get num inputs and outputs of an onnx file'

    onnxModel = onnx.load(onnx_filename)
    inp, out, _ = getIoNodes(onnxModel)
    
    inpShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    outShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    numInputs = 1
    numOutputs = 1

    for n in inpShape:
        numInputs *= n

    for n in outShape:
        numOutputs *= n

def readVnnlib(vnnlibFilename, numInputs, numOutputs):
    '''process in a vnnlib file. You can get numInputs and numOutputs using get_numInputs_outputs().

    this is not a general parser, and assumes files are provided in a 'nice' format. Only a single disjunction
    is allowed

    output a list containing 2-tuples:
        1. input ranges (box), list of pairs for each input variable
        2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output. 
                          Each element in the list is a term in a disjunction for the specification.
    '''

    # example: "(declare-const X_0 Real)"
    regexDeclare = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

    # comparison sub-expression
    # example: "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
    comparisonStr = r"\((<=|>=) (\S+) (\S+)\)"

    # example: "(and (<= Y_0 Y_2)(<= Y_1 Y_2))"
    dnfClauseStr = r"\(and (" + comparisonStr + r")+\)"
    
    # example: "(assert (<= Y_0 Y_1))"
    regexSimpleAssert = re.compile(r"^\(assert " + comparisonStr + r"\)$")

    # disjunctive-normal-form
    # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
    regexDnf = re.compile(r"^\(assert \(or (" + dnfClauseStr + r")+\)\)$")

    rv = [] # list of 3-tuples, (box-dict, mat, rhs)
    rv.append((makeInputBoxDict(numInputs), [], []))
    
    lines = readStatements(vnnlibFilename)

    for line in lines:

        if len(regexDeclare.findall(line)) > 0:
            continue

        groups = regexSimpleAssert.findall(line)

        if groups:
            assert len(groups[0]) == 3, f"groups was {groups}: {line}"
            op, first, second = groups[0]

            for rvTuple in rv:
                updateRvTuple(rvTuple, op, first, second, numInputs, numOutputs)
                
            continue

        ################
        groups = regexDnf.findall(line)

        assert groups, f"failed parsing line: {line}"
        
        tokens = line.replace("(", " ").replace(")", " ").split()
        tokens = tokens[2:] # skip 'assert' and 'or'

        conjuncts = " ".join(tokens).split("and")[1:]

        oldRv = rv
        
        rv = []

       
        for rvTuple in oldRv:
            for c in conjuncts:
                rvTupleCopy = deepcopy(rvTuple)
                rv.append(rvTupleCopy)
                
                cTokens = [s for s in c.split(" ") if len(s) > 0]

                count = len(cTokens) // 3

                for i in range(count):
                    op, first, second = cTokens[3*i:3*(i+1)]

                    updateRvTuple(rvTupleCopy, op, first, second, numInputs, numOutputs)

    # merge elements of rv with the same input spec
    mergedRv = {}

    for rvTuple in rv:
        boxdict = rvTuple[0]
        matrhs = (rvTuple[1], rvTuple[2])

        key = str(boxdict) # merge based on string representation of input box... accurate enough for now

        if key in mergedRv:
            mergedRv[key][1].append(matrhs)
        else:
            mergedRv[key] = (boxdict, [matrhs])

    # finalize objects (convert dicts to lists and lists to np.array)
    finalRv = []

    for rvTuple in mergedRv.values():
        boxDict = rvTuple[0]
        
        box = []

        for d in range(numInputs):
            r = boxDict[d]

            assert r[0] != -np.inf and r[1] != np.inf, f"input X_{d} was unbounded: {r}"
            box.append(r)
            
        specList = []

        for matrhs in rvTuple[1]:
            mat = np.array(matrhs[0], dtype=float)
            rhs = np.array(matrhs[1], dtype=float)
            specList.append((mat, rhs))

        finalRv.append((box, specList))

    #for i, (box, specList) in enumerate(finalRv):
    #    print(f"-----\n{i+1}. {box}\nspec:{specList}")
        
    return finalRv

