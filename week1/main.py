# This code is written by BACHOTTI SAI KRISHNA SHANMUKH
# ROLL NO. EE19B009
import sys
filename = sys.argv[1]
try:
    file_p = open(filename,'r')
except Exception:
    print('Incorrect file name')
# Read lines and close file_p
input_lines = file_p.readlines()
file_p.close()
# Refining to get the required part
circuit_index =[]
end_index =[]
for item in input_lines:
    if '.circuit' in item:
        circuit_index.append(input_lines.index(item))
    if '.end' in item:
        end_index.append(input_lines.index(item))

if len(circuit_index)>1:
    print('Multiple .circuit instructions in the file')
    exit()
if len(end_index)>1:
    print('Multiple .end instructions in the file')
if len(circuit_index) ==0 or len(end_index)==0:
    print('Invalid netlist file')
    exit()
circuit_index = circuit_index[0]
end_index = end_index[0]
ckt_lines = input_lines[circuit_index+1:end_index]

#Removing comments,trailing whitespaces, Reversing the word order and printing from last to first
for i in range(len(ckt_lines)-1,-1,-1):
    ckt_lines[i] = ckt_lines[i].strip('\n')
    cmt_loc = ckt_lines[i].find('#')
    if cmt_loc >=0:
        ckt_lines[i] = ckt_lines[i][:cmt_loc]
    ckt_lines[i] = ckt_lines[i].strip()
    words = ckt_lines[i].split(' ')
    ckt_lines[i] = ' '.join(reversed(words))
    print(ckt_lines[i])
