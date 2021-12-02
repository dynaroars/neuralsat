import numpy as np
import random
import copy

# counting variables
n_iterations=[0]
n_split=[0]
n_update=[0]
n_pivot=[0]
stack_list=[0]

# print the counting variables
TEST_MODE=True 


inf =float('inf'); _inf=float('-inf')
class Reluplex(object):
    def __init__(self, layers, T, relu_T=5):
        """
        Introduction on data structures:
            B: Range of basic variables, NB: Range of non-basic variables
            b: Index of a basic variable nb: Index of a non-basic variable
            tableau: A matrix of size BxNB. Each entry of the matrix has value c_ij, which is the coefficient of the equation relating
            basic variable i with non-basic variable j.
            b_var: An ordered list of the basic variables by key
            nb_var: Same as basic but with non-basic
            variables: A dictionary with all the information of every variable {key: [value, [tableau_position], [nn_architecture_position] }
            [tableau_position] = ['n'|'b', idx] represent the type (nonbasic|basic) equivalent to (col|row) and the position idx
            [nn_architecture_position] = [layer, node, 'n'|'b'|'f'] variable that represent the node number «node» (first node is 0) in «layer» (input layer is 1) of type (normal|basic|function) 
            value_constraints: A dictionary of the value constraints of all the variables so {key: [min, max]}
            relu_constraints: An array of tuples with all the relu relationships [key_n, key_f]
            T is the threshold, usefull to avoid infinite relu pivot loops
        INPUTS: 
            layers: is an ordered list with the weights matrixes that represent the neural network, all the informations about the architecture are extrapolated from the length of layer array
            and from the matrixes' shapes
            inactive: 
        """
        self.layers = layers
        self.tableau, self.b_var, self.nb_var, self.variables, self.value_constraints, self.relu_constraints = self.__create_tableau(layers)
        self.B = range(self.tableau.shape[0])
        self.NB = range(self.tableau.shape[1])
        self.T = T 
        self.relu_T = relu_T
        self.relu_count = np.zeros(len(self.relu_constraints))
        self.list_bounds=[]
        if TEST_MODE: self.stack=0

    def __my_shuffle(self, array):
        #Return a shuffled version of the input array, we use this function to broke the determinism of the scrolling and avoid loops
        # random.shuffle(array) 
        return array 

    def __create_tableau(self, layers):
        #Returns all the needed data structures
        variables={} # { K : [val, ['n'|'b',idx], [layer, node, 'n'|b'|'f']] } #layer 0 if basic #K is an autoincremetal int
        basic_variables=[] #[K, ]
        nonbasic_variables=[] #[K, ]
        value_constraints={} # { K : [min, max] }
        relu_constraints=[] # [[K_n], [K_f]]
        sum=0 #take into account the zero padding
        t=np.array([]).reshape((0,0))
        r=0; c=0; key=0
        #add input var in variables as non-basic
        for v in range(layers[0].shape[0]): #the nn's architecture is encoded in the matrix's shapes
            variables[key]=[0, ['n',c], [1, v, 'n']] #add into the variables dictionary
            value_constraints[key]=[_inf, inf] #set unlimited bounds
            nonbasic_variables.append(key) #add to nonbasic list
            key+=1 #autoincrement int key
            c+=1 #increment col position in the tableau
        for l in range(len(layers)): #for every hidden layer and the output
            h=layers[l].shape[1] #number of nodes in the current layer
            #add all hidden nodes as basic and non basic variables
            for v in range(h):
                variables[key]=[0, ['n',c], [l+2, v, 'n']] 
                nonbasic_variables.append(key)
                value_constraints[key]=[_inf, inf]
                key+=1
                c+=1
                variables[key]= [0, ['b', r], [l+2, v, 'b']]
                basic_variables.append(key)
                value_constraints[key]=[0, 0] #basic variables need to be zero
                key+=1
                r+=1 #basic variables are on the tbleau's row
            if(l!=len(layers)-1): #layer output doesn't need the zeros part because it hasn't the relu duplication
                #if it's not the output layer -> I have also the relu clone hidden f
                for v in range(h):
                    variables[key]=[0, ['n', c], [l+2, v, 'f']]
                    value_constraints[key]=[0, inf] ##relu output (f) nodes are constraints from 0 to inf
                    nonbasic_variables.append(key)
                    relu_constraints.append([key-2*h+v, key]) #I calculate the related node and add their relation to relu_constraints list
                    key+=1
                    c+=1
                t_=np.concatenate([np.zeros((h,sum)), layers[l].T, -np.eye(h), np.zeros((h,h))], axis=1) #create tableau in the form [zero_padding, W.T, -eye, zeros_for_relu_f]
                #sum-=h #output doesn't have the f part so I have to reduce the sum
            else:
                t_=np.concatenate([np.zeros((h,sum)), layers[l].T, -np.eye(h)], axis=1)
            sum=t_.shape[1]-h #calc shift for the next layer
            t=np.concatenate([np.concatenate([t, np.zeros((t.shape[0], t_.shape[1]-t.shape[1]))], axis=1),t_], axis=0) #add the ending zero padding and vertical concatenation
        #I return -t to be similar to the solution by hand, but multiply all the equation by -1 not change the relations
        return (t, basic_variables, nonbasic_variables, variables, value_constraints, relu_constraints)

    def __pivot(self, b, nb, k1, k2):
        if TEST_MODE: n_pivot[0]+=1
      #Switch a basic variables with a non basic one and update all the relations in the tableau
        c = self.tableau[b][nb]
        #See documentation for the matematical formula
        for j in self.NB:
            self.tableau[b][j] = -self.tableau[b][j]/c
        self.tableau[b][nb] = 1/c

        for i in self.B:
            if i == b:
                continue
            c = self.tableau[i][nb]
            for j in self.NB:
                if j == nb:
                    self.tableau[i][j] = c * self.tableau[b][j]
                else:
                    self.tableau[i][j] = c * self.tableau[b][j] + self.tableau[i][j]

        self.b_var[self.b_var.index(k1)], self.nb_var[self.nb_var.index(k2)] = self.nb_var[self.nb_var.index(k2)], self.b_var[self.b_var.index(k1)] #switch the array
        self.variables[k1][1], self.variables[k2][1] = self.variables[k2][1] , self.variables[k1][1] #switch the position in the tableau in variables map

    def __slack(self, b, pos=True): #b is the row of tableu
        # Represent if a variables could encrease or decrease given the relation with the basic one given by input
        #it returns the candidate variable or False
        for var in self.__my_shuffle(self.nb_var):
            j = self.variables[var][1][1]
            if pos:
                if self.tableau[b][j] > 0 and self.variables[var][0] < self.value_constraints[var][1]:
                    return var
                if self.tableau[b][j] < 0 and self.variables[var][0] > self.value_constraints[var][0]:
                    return var
            else:
                if self.tableau[b][j] < 0 and self.variables[var][0] < self.value_constraints[var][1]:
                    return var
                if self.tableau[b][j] > 0 and self.variables[var][0] > self.value_constraints[var][0]:
                    return var
        return False

    def __value_success(self):
        #checks if any variable is out of value bounds, and if it finds one that isn't returns its key and if positive or negative slack is needed, otherwise return True
        for k in self.variables.keys(): #for every variaables
            val=self.variables[k][0]
            if val < self.value_constraints[k][0]:
                return k, True
            if val > self.value_constraints[k][1]:
                return k, False
        return True, None #otherwise I return true

    def __relu_success(self):
        #checks if any relu pair is not satisfied, and if it finds one that isn't returns that pair, otherwise return true
        for relu_pair in self.relu_constraints:
            node_b = relu_pair[0]
            node_f = relu_pair[1]
            if self.variables[node_f][0] == max(0, self.variables[node_b][0]):
                pass
            else:
                return relu_pair
        return True

    def __relu_split(self, relu_pair):
        if TEST_MODE: 
            print('splitting ..................')
            n_split[0]+=1
        """This function is to be called when the treshold for relu updates has been overcomed. It has to make 2 other problems: one in which the neuron that 
        overcame the treshold is active, and one in which is inactive"""
        node_b, node_f = relu_pair

        problem_active = copy.copy(self) #I copy the current instance of the problem
        problem_active.__active_node(relu_pair) #I active the relu constraints
        if TEST_MODE: problem_active.stack=self.stack+1
        problem_unactive = copy.copy(self)
        problem_unactive.__deactivate_node(relu_pair) #I unactive the relu constraints
        if TEST_MODE: problem_unactive.stack=self.stack+1
        if TEST_MODE: stack_list.append(problem_unactive.stack)
        """We need to find a way to make the active and unactive problems behave as we want. Probably that means tunning the create_tableau function to make it able
        to select some neurons not to have the relu relationship. For the active function that neuron will need node_b=node_f and value constraint [0, inf] and for the inactive it has node_b
        with values [-inf, 0] and node_f with values [0, 0], in both cases removing the pair from relu_constraints"""
        if problem_active.new_run(False) is True: #I try to solve the active splitted problem
            return True
        if problem_unactive.new_run(False) is True: #I try to solve the unactive splitted problem
            return True
        return False

    def __active_node(self, relu_pair):
        #make the relu_pair relation active, this means relu constraints mantained and node_b limited from 0 to inf
        node_b=relu_pair[0]
        node_f=relu_pair[1]
        self.value_constraints[node_b]=[0, inf]
        self.value_constraints[node_f]=[0, inf]

    def __deactivate_node(self, relu_pair):
        #for deactivate a relu pair I need to switch off the node_f (set to zero), and make the correspective node_b free to move under the 0 bound. I have to broke their relu relation too
        node_b=relu_pair[0]
        node_f=relu_pair[1]
        self.value_constraints[node_b]=[_inf, 0]
        self.value_constraints[node_f]=[0, 0]
        idx=self.relu_constraints.index(relu_pair)
        self.relu_constraints.remove(relu_pair)
        np.delete(self.relu_count, idx)

    def __update(self, nb): #nb is the index in the array
        if TEST_MODE: n_update[0]+=1
        #Increase/decrease to meet upper bound/lower bound). Update the corresponding basic variables
        var = self.nb_var[nb] #get the key
        value = self.variables[var][0]
        if value < self.value_constraints[var][0]:
            self.variables[var][0] = self.value_constraints[var][0]
        if value > self.value_constraints[var][1]:
            self.variables[var][0] = self.value_constraints[var][1]
        tableau_idx=self.variables[var][1][1] #get the tableau index
        delta = self.variables[var][0] - value
        for b, b_name in enumerate(self.b_var):
            self.variables[b_name][0] = self.variables[b_name][0] + delta * self.tableau[b][tableau_idx]

    def __relu_update(self, relu_pair):
        #Update a pair to meet relu requirements, I update backwards from f to n node
        #I'm here if node_b is nb
        node_b = relu_pair[0]
        node_f = relu_pair[1]
        
        value = self.variables[node_b][0]
        self.variables[node_b][0]=self.variables[node_f][0]
        delta=self.variables[node_b][0] - value
        pos_nodeb=self.variables[node_b][1][1]
        for b, b_name in enumerate(self.b_var):
            self.variables[b_name][0] = self.variables[b_name][0] + delta * self.tableau[b][pos_nodeb]

    def set_bounds_list(self, bounds_list):
      #I can express all the bounds as a list and set them with only one step
      for b in bounds_list:
        self.set_bounds(b[0], b[1])
      return
    
    def new_run(self, debug_mode):
        for t in range(self.T): #T treshold
            k1, slack = self.__value_success() #get key and slack or True
        
            if k1 is True: #if all the values are in their bounds
                relu_pair = self.__relu_success() #try to control the relu relations
                if relu_pair is True: #if all the relu constraints are satisfied too, then my problem is SAT
                    return True #SAT
                #if there is a relu relation not satisfied
                node_b = relu_pair[0] # I consider the node_b
                if node_b in self.nb_var: # Is node_b -nb?
                    self.__relu_update(relu_pair) #I only update the relu_pair backward
                else:
                    #node_b is -b, I can't update it directly
                    row_node_b=self.variables[node_b][1][1] #get the position of node_b in the tableau
                    for nbasic_node in self.__my_shuffle(self.nb_var): #We select any basic variable to pivot, we need shuffle to break the determinism and prevent loops @possible improvement a GLIE choosing strategy
                        row_nbasic_node=self.variables[nbasic_node][1][1] #get the tableu position of the nonbasic node
                        if (self.tableau[row_node_b][row_nbasic_node]!=0): #I can't pivot with a variable not correlated to me, with zero in tableau, because I have a division by zero
                            self.__pivot(row_node_b, row_nbasic_node, node_b, nbasic_node) #I have the two tableau positions so I pivot
                            self.__relu_update(relu_pair) #now node_b is -nb because of pivot, so I update the pair
                            #Relu update could bring to neverending loops, so I need to take count the number of updates
                            relu_idx=self.relu_constraints.index(relu_pair)
                            self.relu_count[relu_idx]+=1 #Count++
                            if (self.relu_count[relu_idx]>self.relu_T): #if count is higher than the relu treashold i split the problem
                                #TO LIMITATE THE RECURSION
                                if self.value_constraints[relu_pair[0]][0]==_inf and self.value_constraints[relu_pair[0]][1]==inf: #If I'm trying to split again my problem is UNSATISFABLE
                                    if self.__relu_split(relu_pair) is True: #Otherwise, if one split is SAT all the problem is SAT
                                        return True
                                    else: #if they are both unsatisfable the problem is unsatisfable
                                        return False
                                else: 
                                    return False
                            break
            else: 
                try:
                    idx1=self.nb_var.index(k1) #try get the index in nb_array of the key1 (only if key1 is -nb)
                    #k1 is -nb
                    self.__update(idx1) #update K1
                except:
                    #k1 is -b
                    row_k1=self.variables[k1][1][1] #get the position of key1 in the tableau
                    k2=self.__slack(row_k1, slack) #apply the slack -> key or False
                    if k2 is False: return False #UNSAT
                    row_k2=self.variables[k2][1][1] #get the tableu position of key2, (not a 'real' row)
                    self.__pivot(row_k1, row_k2, k1, k2) #I have the two tableau positions so I pivot
                    try:
                        idx1=self.nb_var.index(k1) #now I k1 is -nb because of pivot, I try to get the index in the nb_array
                        self.__update(idx1) #update K1
                    except:
                        return 'err' #error
            t+=1 #iteration counter
            if TEST_MODE: n_iterations[0]+=1
            if debug_mode: self.print_debug(t) #report printer for every steps

        return 'err' #too many iterations

    def print_debug(self, step):
        #usefull for debug, it prints important information for every step
        print('step n.',step) #current step
        print(self.tableau) #current tableau
        for key, var in self.variables.items():
            print(key, var[2], var[0],self.value_constraints[key], var[1]) #variables dict in a readible representation

    def set_bounds(self, node, bounds): #node (layer, node)  bounds (min, max)
        # I use this function to set the constraints of my problem
        node.append('n') #I can set only n type nodes!
        try:
            key=[key for key in self.variables if self.variables[key][2] == node][0] #search the key of my node, otherwise I return an error
            node.pop()
            self.list_bounds.append([node, bounds])
        except:
            print('Node is not present in the NN')
            return
        self.value_constraints[key]=bounds #if it's correct I set the bonds

if __name__ == "__main__":
  W1=np.array([1.0, -1.0]).reshape((1,2))
  W2=np.array([1.0, 1.0]).reshape((2,1))
  # W1=np.random.randn(1,500)
  # W2=np.random.randn(500,1)
  #**** HOW TO USE ****#
  #extract the wight matrices into a list, respect the order of the layers
  layers=[W1, W2]
  #create an istance of Reluplex, specifying the max number of steps and optionaly the relu threshold
  test_instance = Reluplex(layers, 150)
  #set all the constraints manually or using a list of them
  test_instance.set_bounds([1, 0],[-1, 10]) #this means node 0 of the layer 1 needs to be between 0 and 1
  # test_instance.set_bounds([3,0],[1.5,2]) #layers start from 1, nodes in layer start from 0, it's possible to set only n nodes
  test_instance.set_bounds([3, 0],[1, 20]) #layers start from 1, nodes in layer start from 0, it's possible to set only n nodes
  #run the verification, with or without (False) the report printing
  print(test_instance.new_run(True)) #this print the outcome
  #this example should be True

