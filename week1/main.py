"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
	EE2703 Assignment 1 Solution
"""
import sys
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
for item in input_lines:
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
if circuit_index[0] > end_index[0]:
    print('Error in netlist file: .end should come after .circuit')
    exit()
ckt_lines = input_lines[circuit_index[0]+1:end_index[0]]
"""
Removing trailing whitespaces and \n, Reversing the word order in a line and printing from last to first line
"""
for i in range(len(ckt_lines)-1,-1,-1):
    ckt_lines[i] = ' '.join(reversed((ckt_lines[i].strip('\n ')).split()))
    if ckt_lines[i] != '':
        print(ckt_lines[i])
    	
