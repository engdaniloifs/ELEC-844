import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import csv
from scipy.spatial import cKDTree
import pandas as pd

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

def find_nearest_node_kdtree(nodes, point,nodes_test,goal, k=50):
    """
    nodes: list/array of (x, y, theta)
    point: (x, y, theta)
    returns: the node from 'nodes' that minimizes your metric(node, point)
             among the k closest in Euclidean (x,y).
    """
    if not nodes:
        return None

    # 1) KD-tree on XY
    xy = nodes_test[:len(nodes), :2]  # extract XY only

    tree = cKDTree(xy)

    

    k = min(k, len(nodes))


    dists, idxs = tree.query(point[:2], k=k)


    idxs = np.atleast_1d(idxs)  # ensure iterable


    # 2) Apply YOUR metric on the k candidates

    best_node = None

    best_cost = float('inf')

    for i in idxs:
        n = nodes[i]
        cost = metric(n, point, goal)  # <-- your metric, unchanged
        if cost < best_cost:
            best_cost, best_node = cost, n

    # if iterations > 2000:
    #     print(f"KD-tree timings (ms): t_1: {t_1*1000:.4f}, t_2: {t_2_end*1000:.4f}, t_3: {t_3_end*1000:.4f}, t_4: {t_4_end*1000:.4f}, t_5: {t_5_end*1000:.4f}, t_6: {t_6_end*1000:.4f}, t_7: {t_7_end*1000:.4f}, t_8: {t_8_end*1000:.4f}")
    #     input("press enter to continue")
    return best_node



def metric(node1,node2,goal,w_angle = 1.5,w_angle_to_go = 0.5):
        if distance(node2, goal) < 0.05 and abs(wrap_to_pi(node2[2] - goal[2])) < np.deg2rad(20):
            w_angle = 10.0
            w_angle_to_go = 10.0
        ang_diff = abs(wrap_to_pi(node1[2] - node2[2]))  # wrap angle
        ang_to_go = abs(wrap_to_pi(node1[2] - np.atan2(node2[1] - node1[1], node2[0] - node1[0])))
        
        return  w_angle * ang_diff + w_angle_to_go * abs(ang_to_go)

def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def steer(nearest, point, step_size,car_parameters,obstacles,goal):
    speed,maximum_steering,L = car_parameters[:3]
    
    
    step_time = step_size/speed
    #input(step_time)
    T = 0
    t = step_time/40
   # input(step_time)
    x_new,y_new,theta_new = nearest
    alpha = theta_new - np.atan2(point[1] - y_new, point[0] - x_new) 
    alpha = wrap_to_pi(alpha)
    
    phi = -np.atan2(alpha*L,speed*step_time)
    phi = np.clip(phi,-maximum_steering,maximum_steering)
    

    while (distance((x_new, y_new, theta_new), point) > (0.02) and (speed*T)<step_size):     
        x_new = x_new + speed*t*np.cos(theta_new)
        y_new = y_new + speed*t*np.sin(theta_new)
        theta_new = wrap_to_pi(theta_new + (speed*t*np.tan(phi))/L)
        
        T += t
        if is_in_obstacle((x_new,y_new,theta_new),obstacles,car_parameters):
            return (x_new, y_new, theta_new), False,False
        if (distance((x_new,y_new,theta_new),goal) < 0.05) and (abs(wrap_to_pi(theta_new - goal[2]))< np.deg2rad(20)):
            return (x_new, y_new, theta_new), True, True
        
        if not (-2 <= x_new <= 2 and -2 <= y_new <= 2):
            return (x_new, y_new, theta_new), False,False
    
    return (x_new, y_new,theta_new), True,False


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
        plt.gca().add_patch(plt.Circle((x_goal, y_goal), 0.05, color='red', alpha=0.15))

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

        nodes = [start]
        nodes_test = np.zeros((30000,2))
        nodes_test[0] = np.array([start[0],start[1]])
        edges = []
        iterations = 0
        
        plt.pause(2)
        goal_region = False
        t_inicial = time.time()
        t_elapsed = 0
        while not goal_region:

            random_node, iterations = sample(X,epsilon,goal,obstacles,iterations,car_parameters)

            nearest_node = find_nearest_node_kdtree(nodes,iterations, random_node,nodes_test,goal)

            new_node,is_edge_valid,goal_region = steer(nearest_node, random_node, step_size,
                                                car_parameters,obstacles,goal)

            if is_edge_valid:
                nodes.append(new_node)
                nodes_test[len(nodes)-1] = np.array([new_node[0],new_node[1]])
                edges.append((nearest_node,new_node))
                plt.plot([nearest_node[0], new_node[0]], [nearest_node[1], new_node[1]], color='black')  
                plt.plot(new_node[0], new_node[1], 'go', markersize=2)  
                plt.quiver(new_node[0], new_node[1], 0.03*np.cos(new_node[2]),
                            0.03*np.sin(new_node[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)

            # if iterations >2000 :
            #     print(f"t_1: {t_1*1000:.4f}, t_2: {t_2*1000:.4f}, t_3: {t_3*1000:.4f}, t_4: {t_4*1000:.4f}, t_5: {t_5*1000:.4f}")
            #     input("press enter to continue")
            t_elapsed = time.time() - t_inicial
        if goal_region:
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
            print("Goal reached!")
            plt.title(f"RRT Finished - iterations: {iterations}")
            print("Iterations:",iterations)
            print("time:",t_elapsed)
            plt.pause(5)
            plt.show()
            return iterations, t_elapsed, True
        else:
            print("path not found within time limit")
            print("Iterations:",iterations)
            print("time:",t_elapsed)
            plt.pause(5)
            plt.show()
            #plt.close()
            return None,None,None#,None,None,None
    
    #without plot on
    
    nodes = [start]
    nodes_test = np.zeros((30000,2))
    nodes_test[0] = np.array([start[0],start[1]])
    edges = []
    iterations = 0
    goal_region = False
    t_inicial = time.time()
    t_elapsed = 0
    while not goal_region and t_elapsed < 60:

        random_node, iterations = sample(X,epsilon,goal,obstacles,iterations,car_parameters)
        nearest_node = find_nearest_node_kdtree(nodes,random_node,nodes_test,goal)
        new_node,is_edge_valid,goal_region = steer(nearest_node, random_node, step_size,car_parameters,obstacles,goal)

        if is_edge_valid:
            nodes.append(new_node)
            nodes_test[len(nodes)-1] = np.array([new_node[0],new_node[1]])
            edges.append((nearest_node,new_node))
            
        t_elapsed = time.time() - t_inicial
    if goal_region:
        print("Goal reached!")
        print("Iterations: "  ,iterations)
        print("time:",t_elapsed)
        return iterations, t_elapsed, True
    else:
        print("path not found within time limit")
        return np.inf, np.inf, False




def main():

    xlim,ylim,thetalim= (-2, 2),(-2,2), (-np.pi, np.pi)

    # start = (-1.5, 0, np.deg2rad(0)) #map1 a single wall
    # goal = (1.5, 0, np.deg2rad(0))

    # start = (-1.5,-1.5, np.deg2rad(0)) #map2 complex
    # goal = (-1.5,1.5, np.deg2rad(180))

    start = (-1.5,0, np.deg2rad(0))
    goal = (0,0,np.deg2rad(180)) #map3 bugtrap


    epsilon = 0.01
    step_size = 0.4
    init_trial_number = 400
    trials_number = 500
    

    X = [xlim,ylim,thetalim]

    # obstacles = [ ((0,0),(0.22,1.5))]    #map1 a single wall
    # obstacles = [((-1.2968538829963083, -0.5840415081573553), (0.44664215576084154, 0.4286360934557055)), 
    #              ((0.13027264796016458, -0.29706589303516573), (0.6399725922732928, 0.4012186776476059)), ((-0.8499273856216552, -0.5789428541291168), (0.5872450110966627, 0.5154658292272251)), 
    #              ((-1.0892193163584944, 0.8163982296877597), (0.5229946891033173, 0.5812293244789292)), ((-0.2115127561535839, 0.12508223792819195), (0.6694316322459648, 0.5231279491834893)), 
    #              ((0.5577927656119408, 1.723496494603246), (0.5501748570854095, 0.4969992177930826))] #map2 complex
    obstacles = [                 #map3 bugtrap
    # Top horizontal wall (0.9 wide, 0.1 thick)
    ((0.00,  0.40), (0.90, 0.10)),

    # Bottom horizontal wall (0.9 wide, 0.1 thick)
    ((0.00, -0.40), (0.90, 0.10)),

    # Left vertical wall (0.1 thick, 0.9 tall)
    ((-0.40, 0.00), (0.10, 0.90)),

    # Right-top small segment (0.1 wide, 0.1 tall)
    ((0.40,  0.40), (0.10, 0.10)),

    # Right-bottom small segment (0.1 wide, 0.1 tall)
    ((0.40, -0.40), (0.10, 0.10)),
]

    iterations_list = np.zeros(trials_number,dtype=float)
    runtime_list = np.zeros(trials_number,dtype=float)
    success_ratio_list = np.zeros(trials_number,dtype=float)
    
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

    

    for trial in range(init_trial_number,trials_number):
        print("Trial number:",trial+1)
        iterations, runtime, success = build_RRT(start,goal,X,obstacles,epsilon,
                                                        step_size,car_parameters, trial,plot = False)
        iterations_list[trial] = iterations
        runtime_list[trial] = runtime
        success_ratio_list[trial] = success

    

    df_trials = pd.DataFrame({
    "Trial": np.arange(len(iterations_list)),
    "Iterations": iterations_list,
    "Runtime (s)": runtime_list,
    "Success": success_ratio_list
})

# Compute final statistics
    iterations_median = np.median(iterations_list)
    runtime_median = np.median(runtime_list)
    success_ratio = np.mean(success_ratio_list)

    # Summary dataframe
    df_summary = pd.DataFrame({
        "Iterations Median": [iterations_median],
        "Runtime Median": [runtime_median],
        "Success Ratio": [success_ratio]
    })

    # Save both sheets in one Excel file:
    text_file = "resultsRRT"+str(init_trial_number)+"-"+str(trials_number-1)+".xlsx"
    with pd.ExcelWriter(text_file) as writer:
        df_trials.to_excel(writer, sheet_name="Trials", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print("Saved to", text_file)
if __name__ == "__main__":
    main()