import numpy as np
#noramlize each node's total link value
def normalized_Link(inlinks, graph, nodeNum):
	for i in range(nodeNum):
		for j in range(nodeNum):
			if graph[j][i] == 1:
				inlinks[i] += 1
	for i in range(nodeNum):
		for j in range(nodeNum):
			if graph[j][i] != 0:
				graph[j][i] = graph[j][i]/inlinks[i]
#simRank algorithm
def simRank (graph, nodeNum):
	sim_value = np.mat(np.identity(nodeNum))
	for i in range(10):
		C = 0.5
		#print (C)
		#print ('graph_T:\n', (graph.T))
		#print ('sim_value:\n', sim_value)
		#print ('graph:\n', graph)
		sim_value_new = C*graph.T*sim_value*graph
		#print('new sim')
		#print(sim_value_new)
		for i in range(nodeNum):
			sim_value_new[i,i] = 1
		sim_value = sim_value_new
	return sim_value

#open file and decide the node numbers according to Readme.txt
print ('Input 1~5 for the corresponding graph1~5:')
selection = ''
while(selection not in ['1', '2', '3', '4', '5']):
	print ('Enter the number within the range:')
	selection = input('graph: ')
	if (selection == '1'):
		nodeNum = 6
		fileName = 'graph_1.txt'
		#print(fileName)
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

file = open(fileName) 
#the initial nodes of different graphs
inlinks = [0]*(nodeNum)
graph = np.zeros((nodeNum, nodeNum))
graph_T = np.zeros((nodeNum, nodeNum))

for line in file:
	line = line.replace('\n', '')
	link = line.split(',')
	#print (link[0], link[1])
	graph[int(link[0])-1][int(link[1])-1] = 1
	#print(graph[int(link[0])][int(link[1])])

normalized_Link(inlinks, graph, nodeNum)
#for line in graph:
#	print (line)
#print('inlinks:',inlinks[1:])
graph = np.mat(graph)
#print (type(graph))
#print ('graph_T')
#print (graph.T)

sim_value = simRank (graph,nodeNum)
print ('Final SimRank value:')
print (sim_value)