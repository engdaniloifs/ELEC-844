import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time


def is_in_obstacle(point,obstacles):
    px, py = point

    for (ox, oy), (w, h) in obstacles:
        dx = px - ox
        dy = py - oy
        if abs(dx) <= w / 2 and abs(dy) <= h / 2:
            return True
    return False

def sample(X, epsilon,goal,obstacles,iterations,random_node=None):
    p = np.random.rand()
    if p < epsilon:
        iterations += 1
        return goal,iterations
    else:
        while random_node is None or is_in_obstacle( random_node,obstacles):
            random_node = (np.random.uniform(*X[0]), np.random.uniform(*X[1]))
            iterations += 1
    return random_node, iterations

def find_nearest_node(nodes, point):
    return min(nodes, key=lambda node: distance(node, point))

def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def step(nearest, point, step_size):
    cost = distance(nearest, point)
    if cost < step_size:
        return point,cost
    angle = np.atan2(point[1] - nearest[1], point[0] - nearest[0])
    x = nearest[0] + step_size * np.cos(angle)
    y = nearest[1] + step_size * np.sin(angle)
    return (x, y),step_size

def is_edge_valid(node1, node2, obstacles):
    steps = 5
    for i in range(1,steps + 1):
        t = i / steps
        x = node1[0] + t * (node2[0] - node1[0])
        y = node1[1] + t * (node2[1] - node1[1])
        if is_in_obstacle((x, y), obstacles):
            return False
    return True

def main():

    xlim,ylim = (0, 100),(0,100)
    start = (25, 50)
    goal = (75, 50)
    epsilon = 0.01
    step_size = 2.5

    gap_width = 25
    obstacles = [((50,50),(10,100-2*gap_width))]
    
    plt.figure(figsize=(5, 5))
    plt.plot(*start, 'go', markersize=5, label="Start")
    plt.plot(*goal, 'ro', markersize=5, label="Goal")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("RRT in progress")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    ax = plt.gca()
    for (ox, oy), (w, h) in obstacles:
        
        rect = patches.Rectangle((ox - w/2, oy - h/2), w, h, color='gray', alpha=0.5)
        
        ax.add_patch(rect)

    X = [xlim, ylim]
    
    nodes = [start]
    edges = []
    solution_length = {start: 0}
    iterations = 0
    while goal not in nodes:
        random_node, iterations = sample(X,epsilon,goal,obstacles,iterations)
        nearest_node = find_nearest_node(nodes,random_node)
        new_node,cost = step(nearest_node, random_node, step_size)
        if is_edge_valid(nearest_node,new_node,obstacles):
            nodes.append(new_node)
            edges.append((nearest_node,new_node))
            cost = solution_length[nearest_node]+ cost
            solution_length[new_node] = cost
            
            plt.plot([nearest_node[0], new_node[0]], [nearest_node[1], new_node[1]], color='black')  
            plt.plot(new_node[0], new_node[1], 'go', markersize=2)  
            
    node = goal
    path = []
    while node != start:
        edge = None
        for (n1, n2) in edges:
            if n2 == node:
                edge = (n1, n2)
                break
        if edge in edges:
            path.append(edge)
        node = n1
        
    tree = [nodes, edges]
    
    for (p, c) in reversed(path):   # go from start → goal
        x1, y1 = p
        x2, y2 = c
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)  
        plt.plot(x1, y1, 'ro', markersize=5)          
    
    #plt.legend()
    #plt.title(f"RRT Concluído - Iterações: {iterations}")
    print("Iterations:",iterations)
    print("Vertices:",len(nodes))
    print("Solution length:", solution_length[goal])
    print("Goal reached!")
    plt.show()

if __name__ == "__main__":
    main()