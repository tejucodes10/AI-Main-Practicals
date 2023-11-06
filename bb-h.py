import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Graph:
    """
    Graph class for initializing and managing a graph.
    
    Attributes:
        graph: Dictionary where keys represent nodes, and values are lists of nodes connected to the key node.
        weight: Dictionary where keys represent nodes, and values are lists of weights corresponding to edges connected to the key node.
        heuristic: Dictionary where keys represent nodes, and values are heuristic values from the source to the goal.
    """

    def __init__(self):
        """
        Initializes the graph, weight, and heuristic dictionaries.
        """
        self.graph = {}
        self.weight = {}
        self.heuristic = {}

    def addEdge(self, o, d, w = 1):
        """
        Adds an edge between two points in the graph.

        Parameters:
            o: Origin/start/current node.
            d: Destination node.
            w: Weight of the edge (default = 1).
        """
        if o not in self.graph:
            self.graph[o] = []
            self.weight[o] = []
            self.heuristic[o] = 100
        if d not in self.graph:
            self.graph[d] = []
            self.weight[d] = []
            self.heuristic[d] = 100
        self.graph[o].append(d)
        self.weight[o].append(w)
        combined = sorted(zip(self.graph[o], self.weight[o]), key=lambda x: x[0])
        self.graph[o], self.weight[o] = map(list, zip(*combined))
        self.graph[d].append(o)
        self.weight[d].append(w)
        combined = sorted(zip(self.graph[d], self.weight[d]), key=lambda x: x[0])
        self.graph[d], self.weight[d] = map(list, zip(*combined))

    def addHeuristics(self, o, h):
        """
        Adds heuristic value to the point mentioned.

        Parameters:
            o: Origin/start/current node.
            h: Heuristic value (default value = 100).
        """
        self.heuristic[o] = h

    def __str__(self):
        """
        Prints the graph, weight and hueristic
        """
        return f"{self.graph}\n{self.weight}\n{self.heuristic}"

class Algorithm:

    """
    This class contains searching techniques that can be used on a Graph.
    Parameters:
        g : graph
        o : origin
        d : destination
        w : weight (default value = 1)
        h : heuristics (default value = 100)
    """

    
    
    def EH(self, g, o, d):
        """
        Branch and Bound algorithm with estimated heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]
        total_path = []

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)

            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        if cost+weight+g.heuristic[current]<=best_cost:
                            # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))

        print(best_path, best_cost)
        return total_path
    

class GraphVisualization:

    def visualize_traversal(self, g, o, d, traversal_algorithm, bw = 1):
        G = nx.Graph()
        for node, neighbors in g.graph.items():
            for neighbor, weight in zip(neighbors, g.weight[node]):
                G.add_edge(node, neighbor, weight=weight)

        if traversal_algorithm.__name__ == "BS":
            paths = traversal_algorithm(g, o, d, bw)
        else:
            paths = traversal_algorithm(g, o, d)
        pos = nx.planar_layout(G)  # You can choose a different layout if you prefer.

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            node_labels = {node: f"{node}\nH:{g.heuristic[node]}" for node in G.nodes()}  # Include heuristic values in node labels
            # Draw the graph
            nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, node_color='lightblue', font_color='black', font_weight='bold',labels = node_labels, ax=ax)
            edge_labels = {(node, neighbor): G[node][neighbor]['weight'] for node, neighbor in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8, ax=ax)

            # Highlight the path up to the current step
            if frame < len(paths):
                path = paths[frame]
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

        ani = FuncAnimation(fig, update, frames=len(paths) + 1, repeat=False, interval=2000)  # Adjust the interval to control animation speed
        plt.show()
    

choice = input("Click Enter to continue with saved graph")
g = Graph()
algo = Algorithm()

if choice == '':
    g.addEdge('P','A',4)
    g.addEdge('P','R',4)
    g.addEdge('P','C',4)
    g.addEdge('A','M',3)
    g.addEdge('C','M',6)
    g.addEdge('R','C',2)
    g.addEdge('R','E',5)
    g.addEdge('C','U',3)
    g.addEdge('U','E',5)
    g.addEdge('M','L',2)
    g.addEdge('E','S',1)
    g.addEdge('U','S',4)
    g.addEdge('S','N',6)
    g.addEdge('U','N',5)
    g.addEdge('L','N',5)
    g.addHeuristics('P',10)
    g.addHeuristics('R',8)
    g.addHeuristics('A',11)
    g.addHeuristics('C',6)
    g.addHeuristics('M',9)
    g.addHeuristics('E',3)
    g.addHeuristics('U',4)
    g.addHeuristics('S',0)
    g.addHeuristics('N',6)
    g.addHeuristics('L',9)
else:
    pass

print(g.graph)

GraphVisualization().visualize_traversal(g, 'P', 'S', algo.EH)





