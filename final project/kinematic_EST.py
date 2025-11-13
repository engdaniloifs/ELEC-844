import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import time
from scipy.spatial import cKDTree

def check_goal_zone(nodes,goal,threshold=0.05):
    for node in nodes:
        dist = distance(node, goal)
        if dist <= threshold:
            #print("Node",node,"is within goal threshold:",dist)
            if abs(wrap_to_pi(node[2] - goal[2]))< np.deg2rad(20):
                input("should stop now")
                return False
    return True




def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def sample_random_node(nodes,new_node,radius_density,iterations,w,flag_obstacle,p,last_max_w,w_line,nodes_test,sum_w_line):
    iterations += 1
    time_to_sample = time.time()
    t_1_end = 0
    t_2_end = 0
    t_3_end = 0
    t_4_end = 0
    t_1_check_end = 0
    if flag_obstacle:
        
        t_1 = time.time()
        changed_w = []
        
        xy = nodes_test[:len(nodes)]
        t_1_check = time.time()
        kdt = cKDTree(xy)
        t_1_check_end = time.time() - t_1_check
        # Find indices of nodes within radius_density
        idxs = kdt.query_ball_point([new_node[0], new_node[1]], r=radius_density)
        
        # For each neighbor, check the heading difference
        last_max_w_changed = False
        t_1_end = time.time() - t_1
        t_2 = time.time()
        
        for i in idxs:
            node1 = nodes[i]
            if abs(wrap_to_pi(node1[2] - new_node[2])) < np.deg2rad(45):
                w[i] += 1000
                if w[i] > last_max_w:
                    last_max_w_changed = True
                    last_max_w = w[i]
                changed_w.append(i)
        t_2_end = time.time() - t_2
        t_3 = time.time()
        if last_max_w_changed:
            sum_w_line = 0
            
            p = []
            
            w_line = []
            
            for i in range(len(nodes)):
                w_line_value = last_max_w + 1 - w[i]       # assuming w is a list/array
                w_line.append(w_line_value)
                sum_w_line += w_line_value
            # for i in range(len(nodes)):
                # p.append(w_line[i] / sum_w_line)
            t_3_end = time.time() - t_3
        else:
            t_4 = time.time()
            for i in changed_w:
                old_wl = w_line[i]
                new_wl = last_max_w + 1 - w[i]
                w_line[i] = new_wl
                sum_w_line += (new_wl - old_wl)  
            # for i in changed_w:
                # p[i] = w_line[i] / sum_w_line
            t_4_end = time.time() - t_4
        if iterations > 1000 :
            time_to_sample_end = time.time() - time_to_sample
            
            #print("Time to sample weights (s):", time_to_sample_end)
            #input("Press Enter to continue...")
        
    t_5 = time.time()
    selected_node = random.choices(nodes, weights=w_line, k=1)[0]
    t_5_end = time.time() - t_5
    time_to_sample_end = time.time() - time_to_sample
    # if iterations > 4000:
    #     print("iterations:", iterations)
    #     print("Time1", t_1_end*1000,"t_1check",t_1_check_end*1000 ,"Time2", t_2_end*1000, "Time3", t_3_end*1000, "Time4", t_4_end*1000, "Time5", t_5_end*1000)
    #     print("Time to sample total (s):", time_to_sample_end*1000)
    #     input("Press Enter to continue...")
    return selected_node,iterations,w, p,last_max_w,w_line,sum_w_line



def forward_propagate(state, control_input, step_time,car_track_length,obstacles,car_size):
    x, y, theta = state
    v = 0.1 
    delta = control_input
    L = car_track_length
    #print("Forward propagate from:",state,"with steering:",delta)
    #time.sleep(1)
    little_step = step_time / 20
    time_elapsed = time.time()
    for _ in range(10):
        x_new = x + v * np.cos(theta) * little_step
        y_new = y + v * np.sin(theta) * little_step
        theta_new = theta + (v / L) * np.tan(delta) * little_step
        theta_new = wrap_to_pi(theta_new)
        x, y, theta = x_new, y_new, theta_new
        if is_in_obstacle((x_new,y_new,theta_new),obstacles,car_size):
            return (x_new, y_new, theta_new), False
        if not (-2 <= x_new <= 2 and -2 <= y_new <= 2):
            return (x_new, y_new, theta_new), False
    time_elapsed = time.time() - time_elapsed
    
    #input("Time for forward propagate (s):"+str(time_elapsed))
    return (x, y, theta), True

def is_in_obstacle(point,obstacles,car_size = None):
    px, py, ptheta = point
    
    for (ox, oy), (w, h) in obstacles:
        dx = px - ox
        dy = py - oy
        if car_size:
            width_car, length_car = car_size
            c = abs(np.cos(ptheta))
            s = abs(np.sin(ptheta))
            obstacle_check_x = w/2 + (width_car/2)*c + (length_car/2)*s
            obstacle_check_y = h/2 + (width_car/2)*s + (length_car/2)*c
        else:
            obstacle_check_x = w/2
            obstacle_check_y = h/2
        if abs(dx) <= obstacle_check_x and abs(dy) <= obstacle_check_y:
            return True
    return False

def build_EST(start,goal,X,obstacles,radius_density, car_size, step_time,L,trial = 0,plot = False):
    nodes = [start]
    if plot:
        x_start,y_start, theta_start = start 
        x_goal,y_goal,theta_goal = goal
        plt.figure(figsize=(10, 10))
        plt.plot(x_start,y_start, 'bo', markersize=7, label="Start")
        plt.quiver(x_start, y_start, 0.03*np.cos(theta_start), 0.03*np.sin(theta_start), angles='xy', scale_units='xy', scale=0.3, width=0.003)
        plt.plot(x_goal,y_goal, 'ro', markersize=7, label="Goal")
        plt.quiver(x_goal, y_goal, 0.03*np.cos(theta_goal), 0.03*np.sin(theta_goal), angles='xy', scale_units='xy', scale=0.3, width=0.003)
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

        #plt.show()
        nodes = [start]
        nodes_test = np.zeros((20000,2))
        nodes_test[0] = np.array([start[0],start[1]])
        edges = []
        
        solution_length = {start: 0}

        iterations = 0
        #plt.pause(0.1)
        x_new = start
        w = [0]
        discrete_steerings = [-0.6, 0, 0.6]
        free_of_obstacles = True
        
        p = {}
        last_max_w = 0
        w_line = {}
        t_inicial = time.time()
        sum_w_line = 0
        reset = True
        while check_goal_zone(nodes,goal):
            #if reset:
             #   t_inicial = time.time()
              #  reset = False
            t_1 = time.time()
            if iterations % 1000 == 0 and iterations > 0:
                 t_elapsed = time.time() - t_inicial
                 print("Iterations:",iterations)
                 print("Elapsed time (s):", t_elapsed) 
            #     reset = True
            #if iterations % 2000 == 0 and iterations > 0:
             #   print("Iterations:",iterations)
              #  t_elapsed = time.time() - t_inicial
               # print("Elapsed time (s):", t_elapsed) 
                #input("Press Enter to continue...")
    
            v_src,iterations,w,p,last_max_w,w_line,sum_w_line = sample_random_node(nodes,x_new,
                                                                                   radius_density,iterations,w,
                                                                                   free_of_obstacles,p,
                                                                                   last_max_w,w_line,nodes_test,sum_w_line)
            t_1_end = time.time() - t_1
            t_2 = time.time()
            control_input = np.random.choice(discrete_steerings)
            t_2_end = time.time() - t_2
            t_3 = time.time()
            x_new,free_of_obstacles = forward_propagate(v_src, control_input, step_time,L,obstacles,car_size)
            t_3_end = time.time() - t_3
            t_4 = time.time()
            if free_of_obstacles:
                nodes.append(x_new)
                nodes_test[len(nodes)-1] = [x_new[0],x_new[1]]
                w.append(0) 
                w_line.append(0)
                edges.append((v_src, x_new))
                solution_length[x_new] = solution_length[v_src] + step_time * 0.2
                
                #if plot:
                    #plt.plot([v_src[0], x_new[0]], [v_src[1], x_new[1]], color='black')  
                    #plt.plot(x_new[0], x_new[1], 'go', markersize=2)  
                    #plt.quiver(x_new[0], x_new[1], 0.03*np.cos(x_new[2]),
                     #           0.03*np.sin(x_new[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)
                    #plt.pause(0.1)
            t_4_end = time.time() - t_4
            t_loop_end = time.time() - t_1
            # if iterations >4000:
            #     print(f"Time sampling: {1000*t_1_end}, Time control input: {1000*t_2_end}, Time forward propagate: {1000*t_3_end}, Time adding node: {1000*t_4_end}")
            #     print("Total loop time:", 1000*t_loop_end)
            #     input("Press Enter to continue...")
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
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)  
        path_demo.append(goal)
        plt.pause(1)
        plt.title(f"EST Finished - Trial number {trial+1}")
        plt.show()
        print("Iterations:",iterations)
        print("Vertices:",len(nodes))
        print("Solution length:", solution_length[goal])
        
        return iterations, len(nodes), solution_length[goal], path_demo
                    



def main():

    xlim,ylim,thetalim = (-2, 2),(-2,2), (-np.pi, np.pi)
    start = (-1.5, 0, np.deg2rad(0))
    goal = (0.75, 0, np.deg2rad(0))
    control_limits = [(0,1), (-0.5,0,0.5)] # speed, steering angle
    
    step_size = 0.5
    radius_density = step_size/2
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


    iterations, vertices, solution_length, = build_EST(start,goal,X,obstacles,radius_density,
                                                        car_size,step_time, L,trial = 0,plot = True)

if __name__ == "__main__":
    main()