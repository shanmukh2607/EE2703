"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
	EE2703 Assignment 2 Solution
    SPICE Program
"""
import sys
import numpy as np

"""
The filename (or path) is given as a command line argument which can be stored into a list sys.argv. Incase more than one filename or no filename is given, it gives an expected usage message"
"""

if len(sys.argv)!=2 :
    print('\nExpected Usage: %s <inputfile>' % sys.argv[0])
    exit()
filename = sys.argv[1]


"""
The user might input a wrong file and the program gives a customized error message
"""	

try:
    file_p = open(filename,'r')
except IOError:
    print('Invalid File')
    exit()	
    
input_lines = file_p.readlines()	#readlines extracts all lines in the file into a list
file_p.close()				#after reading lines close file_p


"""
Remove the commented lines, extract out the information between .circuit and .end, raise an error if not in right format
"""


for i in range(len(input_lines)):
    cmt_loc = input_lines[i].find('#')	# index of '#' is stored in cmt_loc and is negative if not found
    if cmt_loc >=0:
        input_lines[i] = input_lines[i][:cmt_loc]
circuit_index =[]
end_index =[]
ac_info = None
for item in input_lines:
    if '.ac' == item[0:3]:	# read in if .ac command is put in code
        ac_info  = item.split()
    if '.circuit' == item[0:8]:	# .circuit must be first 8 chars of a line
        circuit_index.append(input_lines.index(item))
    if '.end' == item[0:4]:
        end_index.append(input_lines.index(item))
if len(circuit_index)>1:	#more than one uncommented '.circuit' in the file
    print('Multiple .circuit instructions in the file')
    exit()
if len(end_index)>1:		#multiple uncommented '.end' in the file
    print('Multiple .end instructions in the file')
    exit()
if len(circuit_index) ==0 or len(end_index)==0:		#no '.circuit' or '.end' in the file
    print('Invalid netlist file')
    exit()
if circuit_index[0] > end_index[0]:			#error case of inappropriate order of .circuit and .end
    print('Error in netlist file: .end should come after .circuit')
    exit()
ckt_lines = input_lines[circuit_index[0]+1:end_index[0]]


"""
Removing trailing whitespaces and \n
"""

for i in range(len(ckt_lines)-1,-1,-1):
    ckt_lines[i] = ' '.join((ckt_lines[i].strip('\n ')).split())
    if ckt_lines[i] == '':
        ckt_lines.pop(i)
  
"""
Defining two classes Element and ACElement

This is useful to parse info in the code and store them as attributes.
Name stores the name of the element
Eg: R1 could be the name of a resistor used in circuit

from_node is +ve and to_node is -ve in case of Voltage sources
Passive convention is followed for current 

In class ACElement additional attribute type can be stored by parsing through the lines of code
Eg: When .ac command is used, an additional word 'ac' or 'dc' is provided in the code
"""



class Element:
    type = None
    def __init__(self, line):
        self.name = line[0]
        self.from_node = line[1]
        self.to_node = line[2]
        self.value = float(line[3])

class ACElement:
    def __init__(self,line):
        self.name = line[0]
        self.from_node = line[1]
        self.to_node = line[2]
        self.type = line[3]
        self.value = [float(i) for i in line[4:]]	#stores value for DC and peak2peak voltage + phase for ac

"""

Defining Functions: It's a good practice to write down the function code before the main code

"""        
def M_builder(element):
    """
    M_builder, takes an object (a circuit element in this case) of the above defined classes Element and ACElement and constructs the contribution of that element to M matrix, also called as conductance matrix
    More can be read about this in Modified Nodal Analysis
    """
    a = name2number[element.from_node]		#name2number is a dictionary with name as key and node number as value
    b = name2number[element.to_node]		# This dictionary is written in the later part of the code
    if element.name[0] == 'R':		#If the element is a resistor
        M[a,a] += 1/element.value
        M[b,b] += 1/element.value
        M[a,b] += -1/element.value
        M[b,a] += -1/element.value
    if element.name[0] == 'V':		#If the element is a voltage source
        c = int(element.name[1])
        M[a,node_size+c-1] += 1
        M[b,node_size+c-1] += -1
        M[node_size+c-1,a] += 1
        M[node_size+c-1,b] += -1
    if element.name[0] == 'C':		#If the element is a Capacitor
        M[a,a] += 1j*w*element.value
        M[b,b] += 1j*w*element.value
        M[a,b] += -1j*w*element.value
        M[b,a] += -1j*w*element.value
    if element.name[0] == 'L':		#If the element is an Inductor
        if w!= 0:
            M[a,a] += 1/(1j*w*element.value)
            M[b,b] += 1/(1j*w*element.value)
            M[a,b] += -1/(1j*w*element.value)
            M[b,a] += -1/(1j*w*element.value)
        else :					#In the event of zero frequency an inductor can cause infinite conductance which will blow up the matrix
            zero = 1e-10		# Hence a very small value of 1e-10 as w will be a good approximation to avoid errors
            M[a,a] += 1/(1j*zero*element.value)
            M[b,b] += 1/(1j*zero*element.value)
            M[a,b] += -1/(1j*zero*element.value)
            M[b,a] += -1/(1j*zero*element.value)

def b_builder(element):
    """
    b_builder, takes an object (a circuit element in this case) of the above defined classes Element and ACELement and constructs the contribution of that element to b vector, also called as source vector
    More can be read about this in Modified Nodal Analysis
    """
    fn = name2number[element.from_node]		#name2number is a dictionary with name as key and node number as value
    tn = name2number[element.to_node]		# This dictionary is written in the later part of the code
    
    if element.name[0] == 'V':		# If the element is a voltage source
        x = int(element.name[1])	# x will be useful for indexing 
        if element.type != 'ac':
            b[node_size+x-1,0] += element.value		# For DC source
    
        else:
            b[node_size+x-1,0] += 0.5*element.value[0]*np.exp(1j*w*element.value[1])	# For AC source
            
    if element.name[0] == 'I':		# If the element is a current source
        if element.type != 'ac' :
            b[fn,0] += element.value	# For DC source
            b[tn,0] += -element.value
            
        else :
            b[fn,0] += 0.5*element.value[0]*np.exp(1j*w*element.value[1])	# For AC source
            b[tn,0] += -0.5*element.value[0]*np.exp(1j*w*element.value[1])


"""
Parsing the code lines and collecting required attributes

This segment of code reads through the list of refined lines of code and converts them into objects of the classes defined.
"""
element_list = []		# An empty list which will be used to store all objects of the classes Element and ACElement
node_list = []			# An empty list which will be used to store all node names in the circuit
vs_count = 0			# No of voltage sources
is_count = 0
ac_count = 0			# No of ac sources
for item in ckt_lines:
    item_list = item.split()		# Creates a list of words from a string-line
    
    if item_list[0][0] != 'V' and item_list[0][0]!='I':		# If it's not voltage or current source, read the info into an object of class Element
        element_list.append(Element(item_list))
        
    if item_list[0][0] == 'V':	# For voltage source
        if ac_info!= None:
            element_list.append(ACElement(item_list))	# reads into an object of class ACElement when .ac is used in circuit code (spice)
            if ACElement(item_list).type == 'ac':		# If the type of source is 'ac' then increment ac count
                ac_count +=1
        else :
            element_list.append(Element(item_list))		# reads into an object of class Element when no .ac is used in circuit code  (spice)
        vs_count+=1 # Increment the count if Voltage source
        
    if item_list[0][0] == 'I':	# For current source
        if ac_info!= None:
            element_list.append(ACElement(item_list))	# reads into an object of class ACElement when .ac is used in circuit code (spice)
            if ACElement(item_list).type == 'ac':		# If the type of source is 'ac' then increment ac count
                ac_count +=1
        else :
            element_list.append(Element(item_list))		# # reads into an object of class Element when no .ac is used in circuit code  (spice)
        is_count +=1 # Increment the count if Current source
            
    node_list.extend(item_list[1:3]) # both from and to node names involved in this element are included
    
node_size = len(set(node_list))	# no. of nodes

#NOTE : element_list is now a list with objects of the classes defined in this code

"""
Dictionary to map node names to node numbers
This code snippet creates a dictionary with keys as node names and assigns a node number as value to that key
"""


node_uniquelist = list(set(node_list))	# A list of all node names in circuit (non repetitive)
node_uniquelist.remove('GND')		
node_uniquelist.sort()
name2number = {'GND': 0}		# By default GND will be our node 0
for i in range(len(node_uniquelist)):
    name2number[node_uniquelist[i]] = i+1		# other nodes given numbers starting from 1 to N-1 for N nodes in circuit

"""
Creating np zero arrays for M and b

As we may encounter with complex values we set the dtype as complex
"""

M = np.zeros((node_size+vs_count,node_size+vs_count),dtype = complex)	# shape of M given as input 
b = np.zeros((node_size+vs_count,1),dtype = complex)    		# shape of b given as input

""" 
Ac vs Dc case handling

"""

if ac_info == None :	# if no .ac is used in spice code, w is set to 0, which is trivial
    w = 0

if ac_count ==0 and ac_info!= None:		# if .ac is used but only dc sources are present, w is set to 0 with a message
    w = 0
    print("No AC sources found in the .netlist file, frequency overwritten to 0")

if ac_count!=0 :		# .ac with AC sources in the circuit
    w = 2*np.pi*float(ac_info[-1])
    
"""
Setting M and B by iterating through each object (circuit element of class Element or ACElement)
"""

for obj in element_list:	# iterating through each object in the list of objects
    M_builder(obj) 		#Function
    b_builder(obj)		#Calls
    
"""
Solving for x

We use np.linalg.solve to get the solution of N-1 node voltages for ckt with N nodes
and K currents through K independent voltage sources
"""

x = np.linalg.solve(M[1:,1:],b[1:,])	# Since we predetermine Voltage of GND to be 0, we remove the 0th indexed row and column in M and 0th row in B

"""
Output handling

We use detailed English to output the computed results so that the user can understand
"""

if w==0 :
    output = abs(x.ravel())
    for name,number in name2number.items() :	# iterating through the keys and values of a dictionary to print all node voltages
        if number ==0:
            print('Voltage of node GND : 0V')
        if number < node_size and number >0:
            print('Votlage of node',name,':',output[number-1]) 
    for i in range(1, x.size+2-node_size) :		# iterating through the output solution vector to print all currents in voltage sources
        print("Current passing through Voltage source%d" %i,':',output[i+node_size-2])

if w!= 0:
    output = np.hstack((abs(x),np.angle(x)))

    for name,number in name2number.items() :	# iterating through the keys and values of a dictionary to print all node voltages
        if number ==0:
            print('Voltage of node GND is 0V')
        if number < node_size and number >0:
            print('Votlage of node',name,'is \nmagnitude:',output[number-1][0],'V\tphase:',output[number-1][1])
    for i in range(1, x.size+2-node_size) :		# iterating through the output solution vector to print all currents in voltage sources
        print("Current passing through Voltage source V%d" %i,'is \nmagnitude:',output[i+node_size-2][0],'A\tphase:',output[i+node_size-2][1])
        # Both magnitude and phase are printed for ac circuits
        
# Other circuit information:
print("Other circuit information:")
print("No. of nodes including GND :", node_size)
print("No. of Independent voltage sources in the circuit :", vs_count)
print("NO. of Independent current sources in the circuit :", is_count)
if ac_info != None:
    print("No. of AC Independent sources in the circuit :", ac_count)
