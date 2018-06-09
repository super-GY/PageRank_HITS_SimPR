#PageRank algorithm
def PageRank(data, nodeNum, page_value, outlinks):
	new_page_value = [0]*(nodeNum+1)
	for i in range(10):
		for node in range(1, nodeNum+1):
			for link_node in range(1, nodeNum+1):
				if node != link_node:
					if data[link_node][node] == 1:
						#print ('current links:', outlinks[link_node])
						new_page_value[node] += page_value[link_node]/outlinks[link_node]
			new_page_value[node] += 0.15/nodeNum
		sum_value = sum(new_page_value[1:])
		#print (sum_value)
		for node in range(1, nodeNum+1):
			new_page_value[node] = new_page_value[node] / sum_value
		page_value = new_page_value	
		#print('pagevalue:',page_value)
		new_page_value = [0]*(nodeNum+1)
	return page_value
#count links fo each node's outlink
def countLink(outlinks, graph, nodeNum):
	for i in range(1, nodeNum+1):
		for j in range(1, nodeNum+1):
			if graph[i][j] == 1:
				outlinks[i] += 1

#open file and decide the node numbers according to Readme.txt
print ('Input 1~8 for the corresponding graph1~6, 7 for transaction graph, 8 for assciation rules graph:')
selection = ''
while(selection not in ['1', '2', '3', '4', '5', '6', '7', '8']):
	print ('Enter the number within the range:')
	selection = input('graph: ')
	if (selection == '1'):
		nodeNum = 5
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
page_value = [(1/nodeNum)]*(nodeNum+1)
outlinks = [0]*(nodeNum+1)
graph = [[0]*(nodeNum+1) for i in range(nodeNum+1)]

for line in file:
	line = line.replace('\n', '')
	line = line.replace(' ', ',')
	link = line.split(',')
	#print (link[0], link[1])
	graph[int(link[0])][int(link[1])] = 1
	#print(graph[int(link[0])][int(link[1])])

#print ('outlink:',outlinks[1:])
countLink(outlinks, graph, nodeNum)
#print ('outlink:',outlinks[1:])
#print('graph:')
#for line in graph:
	#print (line)

page_value = PageRank(graph, nodeNum, page_value, outlinks)
fourdigit_page = [ '%.8f' % elem for elem in page_value ]
# for i in range(len(fourdigit_page)-1):
# 	for j in range(len(fourdigit_page)-i-1):
# 		if fourdigit_page[j]<fourdigit_page[j+1]:
# 			fourdigit_page[j],fourdigit_page[j+1]=fourdigit_page[j+1],fourdigit_page[j]

print (fourdigit_page[1:])
# print (fourdigit_page)
print ('Highest PageRank node and its value:')
print ('Node:', fourdigit_page.index(max(fourdigit_page)), ' PageRank Value: ', max(fourdigit_page))
# print (sum(page_value[1:]))