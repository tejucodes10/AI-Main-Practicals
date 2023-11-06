import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

MAX = np.inf
MIN = -np.inf
arr = [8, 7, 3, 9, 9, 8, 2, 4, 1, 8, 8, 9, 9, 9, 3, 4]
lim = int(math.log(len(arr), 2))

G = nx.Graph()
pos = {}
labels = {}

def add_children(graph, parent_index, depth, is_max_node, alpha=None, beta=None):
    if depth == lim:
        return arr[parent_index]
    
    if is_max_node:
        best = MIN
        for i in range(2):
            child_index = parent_index * 2 + i
            child_value = add_children(graph, child_index, depth + 1, False, alpha, beta)
            best = max(best, child_value)
            
            # Create nodes and edges
            graph.add_node(child_index, value=arr[child_index])
            graph.add_edge(parent_index, child_index)
            
            # Positioning and labels for visualization
            pos[child_index] = (depth, -child_index)
            labels[child_index] = arr[child_index]

            if alpha is not None and best >= beta:
                break
            if alpha is not None:
                alpha = max(alpha, best)

        return best
    else:
        best = MAX
        for i in range(2):
            child_index = parent_index * 2 + i
            child_value = add_children(graph, child_index, depth + 1, True, alpha, beta)
            best = min(best, child_value)
            
            # Create nodes and edges
            graph.add_node(child_index, value=arr[child_index])
            graph.add_edge(parent_index, child_index)
            
            # Positioning and labels for visualization
            pos[child_index] = (depth, -child_index)
            labels[child_index] = arr[child_index]

            if beta is not None and best <= alpha:
                break
            if beta is not None:
                beta = min(beta, best)

        return best

# Visualize the alpha-beta pruning tree
add_children(G, 0, 0, True, MIN, MAX)

plt.figure(figsize=(10, 8))
nx.draw(G, pos=pos, labels=labels, with_labels=True, node_color='lightblue', node_size=800, font_size=10)
plt.title("Alpha-Beta Pruning Decision Tree")
plt.show()

# Clear the graph and visualize the min-max tree
G.clear()
pos.clear()
labels.clear()

G.add_node(0, value=arr[0])
add_children(G, 0, 0, True)

plt.figure(figsize=(10, 8))
nx.draw(G, pos=pos, labels=labels, with_labels=True, node_color='lightblue', node_size=800, font_size=10)
plt.title("Min-Max Decision Tree")
plt.show()

