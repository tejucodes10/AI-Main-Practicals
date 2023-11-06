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

    
    def HC(self, g, o, d):
        """
        This implements Hill Climbing on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        path = []
        total_path = []
        visited = set()
        node = o
        while node != d:
            path.append(node)
            visited.add(node)
            neighbors = g.graph[node]
            neighbor_heuristics = [g.heuristic[neighbor] for neighbor in neighbors]
            best_neighbor = neighbors[neighbor_heuristics.index(min(neighbor_heuristics))]
            if best_neighbor in visited:
                return total_path
            node = best_neighbor
            total_path.append(list(path[:]))
        path.append(d)
        total_path.append(list(path[:]))
        print(path)
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

        ani = FuncAnimation(fig, update, frames=len(paths) + 1, repeat=False, interval=3000)  # Adjust the interval to control animation speed
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

GraphVisualization().visualize_traversal(g, 'P', 'S', algo.HC)





