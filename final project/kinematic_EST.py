import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import time
import pandas as pd
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
        if distance((x_new,y_new),goal) < 0.05 and abs(wrap_to_pi(theta_new - goal[2]))< np.deg2rad(20):
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
    np.random.seed(trial)
    random.seed(trial)
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

        iterations = 0
        x_new = start
        w = [0]
        discrete_steerings = (-0.6,0.6)
        free_of_obstacles = True
        last_max_w = 0
        w_line = {}
        plt.pause(0.1)
        t_inicial = time.time()
        flag_goal = False
        t_elapsed = 0
        while not flag_goal and t_elapsed < 30: #:
             
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
                if plot:
                    plt.plot([v_src[0], x_new[0]], [v_src[1], x_new[1]], color='black')  
                    plt.plot(x_new[0], x_new[1], 'go', markersize=2)  
                    plt.quiver(x_new[0], x_new[1], 0.03*np.cos(x_new[2]),
                                0.03*np.sin(x_new[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)
            t_elapsed = time.time() - t_inicial
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
            print("goal reached!")
            print("Iterations:",iterations)
            print("Time:",t_elapsed)

            plt.pause(5)
            plt.show()
        
            return iterations, t_elapsed, True
        else:
            print("path not found within time limit")
            plt.pause(5)
            plt.show()
            return iterations, t_elapsed, False
    else:
        nodes = [start]
        nodes_test = np.zeros((30000,2))
        nodes_test[0] = np.array([start[0],start[1]])
        edges = []

        iterations = 0
        x_new = start
        w = [0]
        discrete_steerings = (-0.6,0.6)
        free_of_obstacles = True
        last_max_w = 0
        w_line = {}
        
        t_inicial = time.time()
        flag_goal = False
        t_elapsed = 0
        while not flag_goal and t_elapsed < 60: #:
            
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
            t_elapsed = time.time() - t_inicial
        if flag_goal:
            print("goal reached!")
            print("Iterations:",iterations)
            print("Time:",t_elapsed)
        
            return iterations, t_elapsed, True
        else:
            print("path not found within time limit")
            return iterations, t_elapsed, False
                    



def main():

    xlim,ylim,thetalim = (-2, 2),(-2,2), (-np.pi, np.pi)
    # start = (-1.5, 0, np.deg2rad(0)) #map1 a single wall
    # goal = (1.5, 0, np.deg2rad(0))

    start = (-1.5,-1.5, np.deg2rad(0)) #map2 complex
    goal = (-1.5,1.5, np.deg2rad(180))

    # start = (-1.5,0, np.deg2rad(0))
    # goal = (0,0,np.deg2rad(180)) #map3 bugtrap


    step_size = 0.4
    radius_density = step_size
    speed = 0.1
    step_time = step_size / speed
    init_trial_number = 0
    trials_number = 100

    X = [xlim,ylim,thetalim]

    # obstacles = [ ((0,0),(0.22,1.5))]    #map1 a single wall

    obstacles = [((-1.2968538829963083, -0.5840415081573553), (0.44664215576084154, 0.4286360934557055)), 
                 ((0.13027264796016458, -0.29706589303516573), (0.6399725922732928, 0.4012186776476059)), ((-0.8499273856216552, -0.5789428541291168), (0.5872450110966627, 0.5154658292272251)), 
                 ((-1.0892193163584944, 0.8163982296877597), (0.5229946891033173, 0.5812293244789292)), ((-0.2115127561535839, 0.12508223792819195), (0.6694316322459648, 0.5231279491834893)), 
                 ((0.5577927656119408, 1.723496494603246), (0.5501748570854095, 0.4969992177930826))] #map2 complex
    
#     obstacles = [                 #map3 bugtrap
#     # Top horizontal wall (0.9 wide, 0.1 thick)
#     ((0.00,  0.40), (0.90, 0.10)),

#     # Bottom horizontal wall (0.9 wide, 0.1 thick)
#     ((0.00, -0.40), (0.90, 0.10)),

#     # Left vertical wall (0.1 thick, 0.9 tall)
#     ((-0.40, 0.00), (0.10, 0.90)),

#     # Right-top small segment (0.1 wide, 0.1 tall)
#     ((0.40,  0.40), (0.10, 0.10)),

#     # Right-bottom small segment (0.1 wide, 0.1 tall)
#     ((0.40, -0.40), (0.10, 0.10)),
# ]
    
    iterations_list = np.zeros(trials_number,dtype=float)
    runtime_list = np.zeros(trials_number,dtype=float)
    success_ratio_list = np.zeros(trials_number,dtype=float)
    
    width =  0.25
    length = 0.45
    L = 0.256 # The vehicle's track length.
    #car_variables = [speed,maximum_steering, L]
    car_size = [width, length]


   
    for trial in range(init_trial_number,trials_number):
        print("Trial number:",trial+1)
        iterations, runtime, success = build_EST(start,goal,X,obstacles,radius_density,
                                                            car_size,step_time, L,trial,plot = False)
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
    text_file = "resultsEST"+str(init_trial_number)+"-"+str(trials_number-1)+".xlsx"
    with pd.ExcelWriter(text_file) as writer:
        df_trials.to_excel(writer, sheet_name="Trials", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        

if __name__ == "__main__":
    main()