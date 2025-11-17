import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import time
from scipy.spatial import cKDTree



def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def sample_random_node(nodes,new_node,radius_density,iterations,w,flag_obstacle,last_max_w,w_line,nodes_test):
    iterations += 1
    if flag_obstacle:
        changed_w = []
        
        xy = nodes_test[:len(nodes)]
        t_1_check = time.time()
        kdt = cKDTree(xy)
        # Find indices of nodes within radius_density
        idxs = kdt.query_ball_point([new_node[0], new_node[1]], r=radius_density)
        # For each neighbor, check the heading difference
        last_max_w_changed = False
        
        for i in idxs:
            node1 = nodes[i]
            if abs(wrap_to_pi(node1[2] - new_node[2])) < np.deg2rad(45):
                w[i] += 1000
                if w[i] > last_max_w:
                    last_max_w_changed = True
                    last_max_w = w[i]
                changed_w.append(i)
        if last_max_w_changed:
            w_line = []
            for i in range(len(nodes)):
                w_line_value = last_max_w + 1 - w[i]       
                w_line.append(w_line_value)
        else:
            for i in changed_w:
                w_line[i] = last_max_w + 1 - w[i]
    
    selected_node = random.choices(nodes, weights=w_line, k=1)[0]
    return selected_node,iterations,w,last_max_w,w_line



def forward_propagate(state, control_input, step_time,car_track_length,obstacles,car_size,goal):
    x, y, theta = state
    v = 0.1 
    delta = control_input
    L = car_track_length
    #print("Forward propagate from:",state,"with steering:",delta)
    #time.sleep(1)
    steps_number = 40
    little_step = step_time / steps_number
    
    time_elapsed = time.time()
    for _ in range(steps_number):
        x_new = x + v * np.cos(theta) * little_step
        y_new = y + v * np.sin(theta) * little_step
        theta_new = theta + (v / L) * np.tan(delta) * little_step
        theta_new = wrap_to_pi(theta_new)
        x, y, theta = x_new, y_new, theta_new
        if is_in_obstacle((x_new,y_new,theta_new),obstacles,car_size):
            return (x_new, y_new, theta_new), False,False
        if not (-2 <= x_new <= 2 and -2 <= y_new <= 2):
            return (x_new, y_new, theta_new), False,False
        if distance((x_new,y_new),goal) < 0.05 and abs(wrap_to_pi(theta_new - goal[2]))< np.deg2rad(30):
            input("Goal reached during forward propagate!")
            return (x_new, y_new, theta_new), True, True
        
        
    
    return (x, y, theta), True,False

def is_in_obstacle(point,obstacles,car_size = None):
    px, py, ptheta = point
    
    for (ox, oy), (w, h) in obstacles:
        dx = px - ox
        dy = py - oy
        if car_size:
            width_car, length_car = car_size
            c = abs(np.cos(ptheta))
            s = abs(np.sin(ptheta))
            obstacle_check_x = w/2 + (length_car/2)*c + (width_car/2)*s
            obstacle_check_y = h/2 + (length_car/2)*s + (width_car/2)*c
        else:
            obstacle_check_x = w/2
            obstacle_check_y = h/2
        if abs(dx) <= obstacle_check_x and abs(dy) <= obstacle_check_y:
            return True
    return False

def build_EST(start,goal,X,obstacles,radius_density, car_size, step_time,L,trial = 0,plot = False):
    nodes = [start]
    np.random.seed(0)
    random.seed(0)
    if plot:
        print("cs")
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
        plt.title("EST in progress")
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
        
        solution_length = {start: 0}

        iterations = 0
        x_new = start
        w = [0]
        discrete_steerings = (-0.6,0.6)
        free_of_obstacles = True
        last_max_w = 0
        w_line = {}
        t_inicial = time.time()
        flag_goal = False
       
        while not flag_goal: #and time.time() - t_inicial < 30:
            
                
           
            v_src,iterations,w,last_max_w,w_line = sample_random_node(nodes,x_new, radius_density,iterations,w,
                                                                                   free_of_obstacles,
                                                                                   last_max_w,w_line,nodes_test)
            control_input = np.random.uniform(*discrete_steerings)
            x_new,free_of_obstacles,flag_goal = forward_propagate(v_src, control_input, step_time,L,obstacles,car_size,goal)
            if free_of_obstacles:
                nodes.append(x_new)
                nodes_test[len(nodes)-1] = [x_new[0],x_new[1]]
                w.append(0) 
                w_line.append(0)
                edges.append((v_src, x_new))
                solution_length[x_new] = solution_length[v_src] + step_time * 0.2
                
                if plot:
                    plt.plot([v_src[0], x_new[0]], [v_src[1], x_new[1]], color='black')  
                    plt.plot(x_new[0], x_new[1], 'go', markersize=2)  
                    plt.quiver(x_new[0], x_new[1], 0.03*np.cos(x_new[2]),
                                0.03*np.sin(x_new[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)
        if flag_goal:
            node = x_new
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
            plt.title(f"EST Finished - iterations {iterations}")
            plt.pause(1)
            
            plt.show()
            print("Iterations:",iterations)
            print("Vertices:",len(nodes))
            # print("Solution length:", solution_length[goal])
        
            return iterations, len(nodes), iterations
        else:
            print("No solution found in time limit.")
            return iterations, len(nodes), float('inf')
                    



def main():

    xlim,ylim,thetalim = (-2, 2),(-2,2), (-np.pi, np.pi)
    start = (-1.2, 0, np.deg2rad(0))
    goal = (1.2, 0, np.deg2rad(0))
    control_limits = [(0,1), (-0.5,0,0.5)] # speed, steering angle
    step_size = 0.4
    radius_density = step_size
    speed = 0.1
    step_time = step_size / speed
    trials_number = 100

    X = [xlim,ylim,thetalim]

    #gap_width = 10
    obstacles = [((0,0),(0.22,1.5))]
    
    iterations_list = np.zeros(trials_number,dtype=float)
    vertices_list = np.zeros(trials_number,dtype=float)
    solution_length_list = np.zeros(trials_number,dtype=float)
    
    width =  0.25
    length = 0.45
    L = 0.256 # The vehicle's track length.
    #car_variables = [speed,maximum_steering, L]
    car_size = [width, length]


   
    for trial in range(trials_number):
        print("Trial number:",trial+1)
        iterations, vertices, solution_length = build_EST(start,goal,X,obstacles,radius_density,
                                                            car_size,step_time, L,trial,plot = True)
        print("Trial",trial+1,"completed.")
        plt.show()
        

if __name__ == "__main__":
    main()