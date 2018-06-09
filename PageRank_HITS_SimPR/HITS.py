import math
#HITS algorithm
def HITS(graph, nodeNum, authority, hub):
	for i in range(1):
		norm = 0
		for node in range(1, nodeNum+1):
			authority[node] = 0
			for vertex in graph:
				nodes_now = vertex.split(',')
				if int(nodes_now[1]) == node:
					authority[node] += hub[int(nodes_now[0])]
			norm += authority[node]
		for node in range(1, nodeNum+1):
			# print('normalization:',authority[node], norm)
			authority[node] = authority[node] / norm
		norm = 0
		for node in range(1, nodeNum+1):
			hub[node] = 0
			for vertex in graph:
				nodes_now = vertex.split(',')
				if int(nodes_now[0]) == node:
					hub[node] += authority[int(nodes_now[1])]
			norm += hub[node]
		for node in range(1, nodeNum+1):
			hub[node] = hub[node] / norm
#open file and decide the node numbers according to Readme.txt
print ('Input 1~8 for the corresponding graph1~6, 7 for transaction graph, 8 for assciation rules graph:')
selection = ''
while(selection not in ['1', '2', '3', '4', '5', '6', '7', '8']):
	print ('Enter the number within the range:')
	selection = input('graph: ')
	if (selection == '1'):
		nodeNum = 5
		fileName = 'graph_1.txt'
	elif (selection == '2'):
		nodeNum = 5
		fileName = 'graph_2.txt'
	elif (selection == '3'):
		nodeNum = 4
		fileName = 'graph_3.txt'
	elif (selection == '4'):
		nodeNum = 7
		fileName = 'graph_4.txt'
	elif (selection == '5'):
		nodeNum = 469
		fileName = 'graph_5.txt'
	elif (selection == '6'):
		nodeNum = 1228
		fileName = 'graph_6.txt'
	elif (selection == '7'):
		nodeNum = 541
		fileName = 'graph_trans.txt'
	elif (selection == '8'):
		nodeNum = 22
		fileName = 'graph_rules.txt'

file = open(fileName) 

#the initial nodes of different graphs
authority = [1]*(nodeNum+1)
hub = [1]*(nodeNum+1)
graph = []

for line in file:
	line = line.replace('\n', '')
	line = line.replace(' ', ',')
	graph.append(line)
#calculate HITS algorithm and print the results
HITS(graph, nodeNum, authority, hub)
#show the results in four digits
fourdigit_auth = [ '%.4f' % elem for elem in authority]
fourdigit_hub = [ '%.4f' % elem for elem in hub]
print ('Authority: \n',fourdigit_auth[1:])
print ('Hub: \n', fourdigit_hub[1:])
#print (authority)
#print (hub)
#print (graph)
file.close()

