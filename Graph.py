from collections import deque
import random


class Node:
    def __init__(self, name):
        self.name = name
        self.adjacent = set()
    
    def addAdjacentNode(self, neighbour):
        self.adjacent.add(neighbour)

    def getAdjacentNodes(self):
        return self.adjacent
    
# Adding nodes
nodeA = Node('a')
nodeB = Node('b')
nodeC = Node('c')
nodeD = Node('d')
nodeE = Node('e')
nodeF = Node('f')
nodeG = Node('g')
# nodeH = Node('h')
nodeA.addAdjacentNode(nodeB)
nodeA.addAdjacentNode(nodeD)
nodeB.addAdjacentNode(nodeC)
nodeB.addAdjacentNode(nodeD)
nodeD.addAdjacentNode(nodeC)
nodeD.addAdjacentNode(nodeE)
nodeD.addAdjacentNode(nodeF)
nodeE.addAdjacentNode(nodeF)
nodeF.addAdjacentNode(nodeG)
# #

class Edge: #Consists of two nodes as objects
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
    
    def __str__(self): #two methods to get readable view of the edge
        return f"Edge({self.node1.name}, {self.node2.name})"
    
    def __repr__(self):
        return self.__str__()

#Adding edges
edgeAB = Edge(nodeA, nodeB)
edgeAD = Edge(nodeA, nodeD)
edgeBC = Edge(nodeB, nodeC)
edgeBD = Edge(nodeB, nodeD)
edgeDC = Edge(nodeD, nodeC)
edgeDE = Edge(nodeD, nodeE)
edgeDF = Edge(nodeD, nodeF)
edgeEF = Edge(nodeE, nodeF)
edgeFG = Edge(nodeF, nodeG)
#

class Graph: #consists of sets of edges and nodes as an objects
    def __init__(self):
        self.edges = set()
        self.nodes = set()
    
    def __str__(self): #object for human-readable access
        node_str = ', '.join(node.name for node in self.nodes)
        edge_str = ', '.join(str(edge) for edge in self.edges)
        return f"Nodes: {node_str}\nEdges: {edge_str}"
    
    def addNode(self, node):#Addding node to the set of nodes
        self.nodes.add(node)

    def addEdge(self, edge): #Add edge adding edge to the set of edges and automatically adding nodes of the edge to the set of nodes
        self.edges.add(edge)
        nodeToAdd1=edge.node1
        nodeToAdd2=edge.node2
        self.nodes.add(nodeToAdd1)
        self.nodes.add(nodeToAdd2)

    def getEdges(self): #get edges end get nodes were used by me vefore setting up the __str__ method. Now, to access everything related to graph print(graph) must be written
        return self.edges
    
    def getNodes(self):
        return self.nodes
    
    def nodeExists(self, node): #Check if the node is in the graph
        return node in self.nodes

#Creating graph and adding edges
graph = Graph()
graph.addEdge(edgeAB)
graph.addEdge(edgeAD)
graph.addEdge(edgeBC)
graph.addEdge(edgeBD)
graph.addEdge(edgeDC)
graph.addEdge(edgeDF)
graph.addEdge(edgeDE)
graph.addEdge(edgeEF)
graph.addEdge(edgeFG)
#

#Graph
print('Graph:\n', graph)
#

class GraphMetrics:
    def __init__(self): #GraphMetrics has no essential attributes to manipulate
        self.allPairs = []
        self.allPaths = []
        self.suitablePaths = [] #suitable paths for calculating betweenness centrality
        self.allPathsDict = {}

    def nodeDegree(self, node): #calculating degree of a node
        degree = len(node.adjacent)
        return degree

    def findShortestPath(self, startNode, targetNode): #method to find the shortest path between two nodes
        if startNode == targetNode: #returns the name of the node incase current node has the common edge with the target node
            return [startNode.name]

        visited = set()  # To keep track of visited nodes
        queue = deque([(startNode, [startNode.name])])  # double ended queue of (current_node, path_list) tuples

        while queue:
            current_node, path = queue.popleft()  # Get the first node in the queue

            visited.add(current_node.name)
            for neighbour in current_node.adjacent:
                if neighbour.name not in visited:
                    if neighbour == targetNode:
                        return path + [neighbour.name]  # Return the path including the target node
                    queue.append((neighbour, path + [neighbour.name]))
                    visited.add(neighbour.name)  # Mark this node as visited

        return []  # Return an empty list if no path is found
    

    def findShortestPaths(self, start_node, end_node): #Used to find all the possible shortest path. This method was added at the end to calculate the most accurate betweennes centrality
        shortest_paths = []
        shortest_length = float('inf')  # Initialize shortest length to positive infinity

        def dfs(node, path, length):
            if node == end_node:
                nonlocal shortest_length
                if length < shortest_length:
                    shortest_length = length
                    shortest_paths.clear()
                if length == shortest_length:
                    shortest_paths.append(path[:])
                return

            for neighbour in node.adjacent:
                if neighbour not in path:
                    dfs(neighbour, path + [neighbour.name], length + 1)

        dfs(start_node, [start_node.name], 0)
        return shortest_paths

    
    def getPathLength(self, node1, node2): 
        path = self.findShortestPath(node1, node2)
        length = len(path)
        if length == 0: #incase there is no path between nodes or finding the path to the node itself
            return length
        else:
            return length - 1 #minus one to count edges instead of nodes
    
    def getTotalLength(self, node, graphic): #Method to calculate the sum of the shortest paths from one specific node to every node in the graph
        if graphic.nodeExists(node):
            total = 0
            for nodeTmp in graphic.nodes:
                length = self.getPathLength(node, nodeTmp)
                total += length
            return total
        
    def getClosenessCentrality(self, node, graphic): #reciprocal of the total length of the shortest paths from specific node to every other node in the graph
        totalLength = self.getTotalLength(node, graphic)
        if totalLength == 0:
            return totalLength #in case node doesn't have outgoing edges
        else: 
            return 1/totalLength
        
    def getAllPairs(self, graphic):
        self.allPairs = []
        for node1 in graphic.nodes:
            for node2 in graphic.nodes:
                if node1 != node2:
                    pair = (node1, node2)
                    self.allPairs.append(pair)
        return self.allPairs
    
    def isPathExist(self, node1, node2):
        return self.getPathLength(node1, node2) > 0
    
    def getAllPaths(self): #every possible shortest path between nodes in the graph
        self.allPaths = []
        for pair in self.allPairs:
            if self.isPathExist(pair[0], pair[1]):
                path = self.findShortestPaths(pair[0], pair[1])
                for option in path:
                    self.allPaths.append(option)
        return self.allPaths
    
    def getAllPathsDict(self): #every possible shortest path between nodes in the graph
        self.allPathsDict = {}
        for pair in self.allPairs:
            if self.isPathExist(pair[0], pair[1]):
                path = self.findShortestPath(pair[0], pair[1])
                self.allPathsDict[pair] = path
        return self.allPathsDict
    
    def getSuitablePaths(self, node): #suitable path from the node perspective to calculate the betweenness centrality
        self.suitablePaths = []
        self.getAllPaths()
        for path in self.allPaths:
            if len(path) > 2 and path[0] != node.name and path[-1] != node.name and node.name in path:
                self.suitablePaths.append(path)
        return self.suitablePaths

    def getBetweennessCentrality(self, node):
        self.getSuitablePaths(node)
        betweennessCentrality = 0
        for path in self.suitablePaths:
            if node.name in path:
                betweennessCentrality += 1
        return betweennessCentrality


#Creating graph metrics
graphMet = GraphMetrics()

#Testing nodeDegree
print('Adjacent nodes of node A:', [node.name for node in nodeA.getAdjacentNodes()])
print('Edges: ', len(graph.edges))
degreeCheck = graphMet.nodeDegree(nodeA)
print('Degree of the node A: ', degreeCheck)
#

#Testing Closeness Centrality
#print('Path: ', graphMet.findPath(nodeD, nodeC, graph, None))
#print('Nodes of the graph: ', [node.name for node in graph.nodes])
#print(len(graphMet.findPath(nodeA, nodeE, graph, None)))
#rint('ShortestPath: '), graphMet.shortestPath(nodeA, nodeE, graph)
print('Shortest Path: ', graphMet.findShortestPath(nodeA, nodeB))
print('Length of the path: ', graphMet.getPathLength(nodeA, nodeB))
print('Total length: ', graphMet.getTotalLength(nodeA, graph))
print('Closeness centrality: ', graphMet.getClosenessCentrality(nodeA, graph))
#

#Testing Betweenness Centrality
def print_all_pairs(pairs):
    pairs_str = ', '.join(f"({pair[0].name}, {pair[1].name})" for pair in pairs)
    print(f"All Pairs: {pairs_str}")

all_pairs = graphMet.getAllPairs(graph)
print_all_pairs(all_pairs)
#print('Path exists: ', graphMet.isPathExist(nodeA, nodeB))
print('All paths: ', graphMet.getAllPaths())
print('Suitable paths: ', graphMet.getSuitablePaths(nodeD))
print('Betweenness Centrality: ', graphMet.getBetweennessCentrality(nodeD))
# print('Shortest paths:')
# pp = graphMet.findShortestPaths(nodeA, nodeC)
# for path in pp:
#     print([node.name for node in path])

graphMet.getAllPathsDict() #creating a dictionairy for avoiding unnecessary calculations

class Agent():
    def __init__(self, start, graphic):
        self.currentNodeObj = start #Starting point of the agent as an object for convenient calculations
        self.currentNodeName = start.name #Starting point of the agent as a string
        self.visited = set()
        self.graphIn = graphic #graph in which the agent is located in

    def getNodeObj(self, nodeName): #getting a node as an object via its name
        for node in self.graphIn.nodes:
            if node.name == nodeName:
                return node

    def getCurrentNode(self): #MEthod is created to check the result that's why it return readable format of the current node
        return self.currentNodeName
    
    def getVisited(self):
        return self.visited

    def randomWalk(self):
        randomPath = random.choice(graphMet.allPaths)
        self.currentNodeName = randomPath[-1] #updating the currentnode
        self.currentNodeObj = self.getNodeObj(randomPath[-1])
        for node in randomPath:
            self.visited.add(node)
    
    def shortestPath(self, node):
        pathToNode = graphMet.allPathsDict[self.currentNodeObj, node]
        self.currentNodeName = node.name
        self.currentNodeObj = node
        for nodeToAddVisited in pathToNode:
            self.visited.add(nodeToAddVisited)

    

#AgentTesting
# agent = Agent(nodeA, graph)
# agent.shortestPath(nodeC)
# agent.randomWalk()
# print('Current node: ', agent.getCurrentNode())
# print('Visited: ', agent.getVisited())
# for pair, path in graphMet.allPathsDict.items():
#     print(pair, "->", path)
#print(agent.getNodeObj('b'))

#Simulation
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
randomNode = random.choice(nodes)
print(randomNode)
for node in graph.nodes:
        if node.name == randomNode:
            startNode = node
agentSimulation = Agent(startNode, graph)
agentSimulation.randomWalk()
agentSimulation.randomWalk()
print('Current node: ', agentSimulation.getCurrentNode())
print('Visited: ', agentSimulation.getVisited())

# Initialize a dictionary to store the frequency of each node
node_frequency = {node.name: 0 for node in graph.nodes}

# Repeat the simulation 1000 times
for _ in range(1000):
    # Reset the agent's position and visited nodes
    agentSimulation = Agent(startNode, graph)
    
    # Perform random walks
    for _ in range(1000):  # Assuming each simulation consists of 1000 steps
        agentSimulation.randomWalk()
    
    # Update the frequency of visited nodes
    visited_nodes = agentSimulation.getVisited()
    for node_name in visited_nodes:
        node_frequency[node_name] += 1

# Calculate the approximate frequency of each node
total_visits = sum(node_frequency.values())
node_frequency_percentage = {node_name: (count / total_visits) * 100 for node_name, count in node_frequency.items()}

# Display the approximate frequency of each node
print("Approximate frequency of each node after 1000 simulations:")
for node_name, frequency in node_frequency_percentage.items():
    print(f"{node_name}: {frequency:.2f}%")