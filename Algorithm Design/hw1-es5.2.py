# - *- coding: utf- 8 - *-
import networkx as nx
from queue import PriorityQueue
import random

boolean = 0
parent = dict()
rank = dict()
def main():
	G=nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6,7,8])
	prince_edge = (1,2)
	G.add_edges_from([(1,2,{'weight':1}),(1,3,{'weight':2}),(2,3,{'weight':2}),(3,6,{'weight':6}),(3,4,{'weight':2}),(4,6,{'weight':8}),(4,5,{'weight':4}),(5,7,{'weight':3}),(6,7,{'weight':4}),(6,8,{'weight':4}),(8,7,{'weight':7}),(2,8,{'weight':3})])
	variable = random.randrange(len(prince_edge))
	visited = set()
	start = prince_edge[variable]
	end = 0
	if (variable==0):
		end = prince_edge[1]
	else:
		end = prince_edge[0]
	visited.add(prince_edge)
	visited.add((prince_edge[1], prince_edge[0]))
	prince_weight = G.get_edge_data(prince_edge[0],prince_edge[1]).get('weight')
	print("start: "+str(start))
	print("end: "+str(end))
	dfs_cicle(G, visited, start, prince_weight, end)
	
	if (boolean): 
		print ("There is no MST containing prince_edge in graph G")
		
	else:
		q = PriorityQueue() 
		list_edge = list(G.edges())

		if prince_edge not in list_edge:
			prince_edge_new = (prince_edge[1], prince_edge[0])
			list_edge.pop(list_edge.index(prince_edge_new))
		else :
			
			list_edge.pop(list_edge.index(prince_edge))
		
		#After checking that there exists a MST containing prince_edge, I begin Kruskal starting from it
		q.put((0, prince_edge))
		for i in range(len(list_edge)):
			
			
			q.put((G.get_edge_data(list_edge[i][0],list_edge[i][1]).get('weight'), list_edge[i]))
			
		
			 
		boo = 0		
		results = kruskal(G, q, boo, prince_edge)
		
		if results[1] :
			print ('the selected edge is in the MST')
			print ( str(results[0]))
		
	
def make_set(vertice):
    	parent[vertice] = vertice
    	rank[vertice] = 0

def find(vertice):
    	if parent[vertice] != vertice:
        	parent[vertice] = find(parent[vertice])
    	return parent[vertice]

def union(vertice1, vertice2):
	root1 = find(vertice1)
	root2 = find(vertice2)
	if root1 != root2:
		if rank[root1] > rank[root2]:
			parent[root2] = root1
		else:
	    		parent[root1] = root2
	if rank[root1] == rank[root2]: rank[root2] += 1
	
def kruskal(G, q, boo, prince_edge):
	minimum_spanning_tree = set()	
	for vertice in G.nodes():
		make_set(vertice)
	while not q.empty():
		aux = q.get()
		vertice1, vertice2 = aux[1]
		if find(vertice1) != find(vertice2):
			union(vertice1, vertice2)
			minimum_spanning_tree.add(aux[1])
			if (aux[1] == prince_edge):
				boo = 1
	    
	return (sorted(minimum_spanning_tree), boo)
def dfs_cicle(G, visited, n, prince_weight, end):
	#This DFS visits each node one time only, considering only the edges with a cost less than prince_weight
	for e in G.edges(n):
		if G.get_edge_data(e[0],e[1]).get('weight') < prince_weight:
				#It seems that networkx considers edges as directed, so every time I visit one of them, I add to visited set its specular too
				visited.add(e)
				visited.add((e[1],e[0]))

				opposite = 0
				if n == e[0]:
					opposite = e[1]
				else:
					opposite = e[0]
				
				if opposite == end:
					global boolean
					boolean = 1
					return True
				else: 
					if opposite not in visited:
						visited.add(opposite)
						dfs_cicle(G, visited, opposite, prince_weight, end)
	return False
main()
