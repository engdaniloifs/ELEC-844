import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import csv
from scipy.spatial import cKDTree

def is_in_obstacle(point,obstacles,car_parameters):
    px, py, ptheta = point
    width_car = car_parameters[3]
    length_car = car_parameters[4]
    
    for (ox, oy), (w, h) in obstacles:
        dx = px - ox
        dy = py - oy
        
        c = abs(np.cos(ptheta))
        s = abs(np.sin(ptheta))
        obstacle_check_x = w/2 + (length_car/2)*c + (width_car/2)*s
        obstacle_check_y = h/2 + (length_car/2)*s + (width_car/2)*c
        
        if abs(dx) <= obstacle_check_x and abs(dy) <= obstacle_check_y:
            return True
    return False

def sample(X, epsilon,goal,obstacles,iterations,car_parameters,random_node=None):
    p = np.random.rand()
    inside_obstacle = True
    if p < epsilon:
        iterations += 1
        return goal,iterations
    else:
        while random_node is None or inside_obstacle:
            random_node = (np.random.uniform(*X[0]), np.random.uniform(*X[1]),np.random.uniform(*X[2]))
            
            inside_obstacle = is_in_obstacle(random_node,obstacles,car_parameters)
            iterations += 1
       
    return random_node, iterations

def find_nearest_node_kdtree(nodes, point, k=100):
    """
    nodes: list/array of (x, y, theta)
    point: (x, y, theta)
    returns: the node from 'nodes' that minimizes your metric(node, point)
             among the k closest in Euclidean (x,y).
    """
    if not nodes:
        return None

    # 1) KD-tree on XY
    xy = np.asarray([n[:2] for n in nodes], dtype=float)
    tree = cKDTree(xy)

    k = min(k, len(nodes))
    dists, idxs = tree.query(point[:2], k=k)
    idxs = np.atleast_1d(idxs)  # ensure iterable

    # 2) Apply YOUR metric on the k candidates
    best_node = None
    best_cost = float('inf')
    for i in idxs:
        n = nodes[i]
        cost = metric(n, point)  # <-- your metric, unchanged
        if cost < best_cost:
            best_cost, best_node = cost, n

    return best_node



def metric(node1,node2,w_angle = 1.5,w_angle_to_go = 0.5):
        if node2 == (1.2, 0, np.deg2rad(0)):
            w_angle_to_go = 10
            w_angle = 10
        #pos_diff = distance(node1[:2], node2[:2])
        ang_diff = abs(np.arctan2(np.sin(node1[2]-node2[2]), np.cos(node1[2]-node2[2])))  # wrap angle
        ang_to_go = wrap_to_pi(node1[2] - np.atan2(node2[1] - node1[1], node2[0] - node1[0]))
        
        return  + w_angle * ang_diff + w_angle_to_go * abs(ang_to_go)

def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def steer(nearest, point, step_size,car_parameters,obstacles,goal):
    speed,maximum_steering,L = car_parameters[:3]
    
    
    step_time = step_size/speed
    #input(step_time)
    T = 0
    t = step_time/10
   # input(step_time)
    x_new,y_new,theta_new = nearest

    while (distance(nearest, point) > (speed*T) and (speed*T)<step_size):
        alpha = theta_new - np.atan2(point[1] - y_new, point[0] - x_new) 
        alpha = wrap_to_pi(alpha)
    
        phi = -np.atan2(alpha*L,speed*step_time)

        phi = np.clip(phi,-maximum_steering,maximum_steering)
        
        x_new = x_new + speed*t*np.cos(theta_new)
        y_new = y_new + speed*t*np.sin(theta_new)
        theta_new = wrap_to_pi(theta_new + (speed*t*np.tan(phi))/L)

        

        if is_in_obstacle((x_new,y_new,theta_new),obstacles,car_parameters):
            return (x_new, y_new, theta_new),step_size, False,False
        T += t
    
        if (distance((x_new,y_new,theta_new),goal) < step_size/2) and (abs(wrap_to_pi(theta_new - goal[2]))< np.deg2rad(20)):
            input("goal reached")
            return (x_new, y_new, theta_new), distance(nearest, (x_new,y_new,theta_new)), True, True
        if not (-2 <= x_new <= 2 and -2 <= y_new <= 2):
            return (x_new, y_new, theta_new), False, False,False

    return (x_new, y_new,theta_new), step_size, True,False


def  build_RRT(start,goal,X,obstacles,epsilon,step_size,car_parameters,trial,plot = False):
    # with plot on
    np.random.seed(trial)
    if plot:
        x_start,y_start, theta_start = start 
        x_goal,y_goal,theta_goal = goal
        plt.figure(figsize=(10, 10))
        plt.plot(x_start,y_start, 'bo', markersize=7, label="Start")
        plt.quiver(x_start, y_start, 0.03*np.cos(theta_start), 0.03*np.sin(theta_start), angles='xy', scale_units='xy', scale=0.3, width=0.003)
        plt.plot(x_goal,y_goal, 'ro', markersize=7, label="Goal")
        plt.quiver(x_goal, y_goal, 0.03*np.cos(theta_goal), 0.03*np.sin(theta_goal), angles='xy', scale_units='xy', scale=0.3, width=0.003)
        plt.gca().add_patch(plt.Circle((x_goal, y_goal), step_size/2, color='red', alpha=0.15))

# main heading arrow
        plt.quiver(x_goal, y_goal, 0.03*np.cos(theta_goal), 0.03*np.sin(theta_goal),
                angles='xy', scale_units='xy', scale=0.3, color='r', width=0.003)

        # two limit arrows (± tolerance)
        for sign in (-1, 1):
            plt.quiver(x_goal, y_goal, 0.03*np.cos(theta_goal + sign*np.deg2rad(20)),
                    0.03*np.sin(theta_goal + sign*np.deg2rad(20)),
                    angles='xy', scale_units='xy', scale=0.3, color='r', alpha=0.6, width=0.003)
        plt.xlim(X[0])
        plt.ylim(X[1])
        plt.title("RRT in progress")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        
        ax = plt.gca()
        for (ox, oy), (w, h) in obstacles:
            
            rect = patches.Rectangle((ox - w/2, oy - h/2), w, h, color='gray')
            
            ax.add_patch(rect)

        #plt.show()
        nodes = [start]
        #nodes_demo = [(start, 0)]
        edges = []
        #edges_demo = []
        solution_length = {start: 0}
        iterations = 0
        t_inicial = time.time()
        plt.pause(0.1)
        #input("Press Enter to start RRT...")
        goal_region = False
        while goal_region == False:
            #print("iteration:",iterations)
            #print("Iterations:",iterations)
            if iterations % 1000 == 0:
                print("Iterations:",iterations)
                t_elapsed = time.time() - t_inicial
                print("Elapsed time (s):", t_elapsed)
            
            if (iterations % 1000 == 0) and iterations != 0:
                plt.title(f"RRT in progress - Iteration {iterations}")
                #
                #input("Press Enter to continue...")
            
            random_node, iterations = sample(X,epsilon,goal,obstacles,iterations,car_parameters)
            nearest_node = find_nearest_node_kdtree(nodes,random_node)
            new_node,cost,is_edge_valid,goal_region = steer(nearest_node, random_node, step_size,
                                                car_parameters,obstacles,goal)
            if is_edge_valid:
                #t_elapsed = time.time() - t_inicial

                nodes.append(new_node)
                #nodes_demo.append((new_node,t_elapsed))
                edges.append((nearest_node,new_node))
                #edges_demo.append(((nearest_node,new_node),t_elapsed))
                
                cost = solution_length[nearest_node]+ cost
                solution_length[new_node] = cost
                
                plt.plot([nearest_node[0], new_node[0]], [nearest_node[1], new_node[1]], color='black')  
                plt.plot(new_node[0], new_node[1], 'go', markersize=2)  
                plt.quiver(new_node[0], new_node[1], 0.03*np.cos(new_node[2]),
                            0.03*np.sin(new_node[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)
            if iterations >15000:
                print("path not found ")
                #plt.close()
                return None,None,None#,None,None,None
                

        node = new_node
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
        path_demo = []
        for (p, c) in reversed(path):   # go from start → goal
            x1, y1,theta1 = p
            x2, y2,theta2 = c
            path_demo.append((x1,y1,theta1))
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=3)  
        path_demo.append(goal)
        
        plt.title(f"RRT Finished - iterations: {iterations}")
        print("Iterations:",iterations)
        print("Vertices:",len(nodes))
#        print("Solution length:", solution_length[goal])
        plt.pause(5)
        plt.show()
        return iterations, len(nodes), solution_length[goal]#, nodes_demo,edges_demo, path_demo
    
    #without plot on
    
    nodes = [start]
    edges = []
    solution_length = {start: 0}
    iterations = 0

    while goal not in nodes:

        random_node, iterations = sample(X,epsilon,goal,obstacles,iterations)
        nearest_node = find_nearest_node(nodes,random_node)
        new_node,cost = steer(nearest_node, random_node, step_size)

        if is_edge_valid(nearest_node,new_node,obstacles):

            nodes.append(new_node)
            edges.append((nearest_node,new_node))
            cost = solution_length[nearest_node]+ cost
            solution_length[new_node] = cost

    print("Iterations:",iterations)
    print("Vertices:",len(nodes))
    print("Solution length:", solution_length[goal])

    return iterations, len(nodes), solution_length[goal]
    


def main():

    xlim,ylim,thetalim= (-2, 2),(-2,2), (-np.pi, np.pi)
    start = (-1.2, 0, np.deg2rad(0))
    goal = (1.2, 0, np.deg2rad(0))
    epsilon = 0.01
    step_size = 0.1
    trials_number = 200

    X = [xlim,ylim,thetalim]

    obstacles = [((0,0),(0.12,1.3))]
    
    iterations_list = np.zeros(trials_number,dtype=float)
    vertices_list = np.zeros(trials_number,dtype=float)
    solution_length_list = np.zeros(trials_number,dtype=float)
    
    #car_variables
    maximum_steering = 0.6 #rad
    speed = 0.1 #
    width =  0.25
    length = 0.45
    L = 0.256 # The vehicle's track length.
    car_parameters = [speed,maximum_steering, L,width,length]

   # iterations, vertices, solution_length, nodes_demo, edges_demo, path_demo = build_RRT(start,goal,X,obstacles,epsilon,
    #                                                    step_size,car_parameters, trial = 0,plot = True)
    
#     #with open("tree.csv", mode="w", newline="") as file:
#      #   writer = csv.writer(file)
        
#         # Write headers
#       #  writer.writerow(["node sampled, edge_created"])
#        # writer.writerow([
#             "x", "y", "theta", "time",        # node sampled
#             "x1", "y1", "theta1",             # edge start
#             "x2", "y2", "theta2", "time"        # edge end/time
#             ])
    
#     # Write each node and corresponding edge
#         for ((x, y, theta), t), ((n1, n2), t_elapsed) in zip(nodes_demo, edges_demo):
#             (x1, y1, theta1) = n1
#             (x2, y2, theta2) = n2
#             writer.writerow([x, y, theta, t, x1, y1, theta1, x2, y2, theta2, t_elapsed])

# # Separate with open (same indentation)
#     with open("path.csv", mode="w", newline="") as file:
#         writer = csv.writer(file)
        
#         # Write headers
#         writer.writerow(["x", "y", "theta"])
        
#         for (x, y, theta) in path_demo:
#             writer.writerow([x, y, theta])
        
    
#     iterations_list[0] = iterations
#     vertices_list[0] = vertices
#     solution_length_list[0] = solution_length

    

    for trial in range(1,trials_number):
        print("Trial number:",trial+1)
        iterations, vertices, solution_length = build_RRT(start,goal,X,obstacles,epsilon,
                                                        step_size,car_parameters, trial,plot = True)
    


    iterations_mean = np.mean(iterations_list)
    vertices_mean = np.mean(vertices_list)
    solution_length_mean = np.mean(solution_length_list)

    print("Iterations mean:",iterations_mean)
    print("Vertices mean:",vertices_mean)
    print("Solution length mean:", solution_length_mean)


    plt.show()

if __name__ == "__main__":
    main()