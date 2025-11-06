import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import time

def check_goal_zone(nodes,goal,threshold=0.2):
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

def sample_random_node(nodes,new_node,radius_density,iterations,w,flag_obstacle):
    iterations += 1
    w_line = {}
    
    p = {}
    if flag_obstacle:
        for node1 in nodes:
            if distance(node1,new_node) < radius_density:
                if abs(wrap_to_pi(node1[2] - new_node[2]))< np.deg2rad(15):
                    w[node1] +=  1
    max_w = max(w.values())
    for node in nodes:
        w_line[node] = max_w + 1 - w[node]
    sum_w_line = sum(w_line.values())
    for node in nodes:
        p[node] = w_line[node] / sum_w_line
    selected_node = random.choices(nodes, weights=[p[node] for node in nodes], k=1)[0]
    return selected_node,iterations,w



def forward_propagate(state, control_input, step_time,car_track_length,obstacles):
    x, y, theta = state
    v = 0.2 
    delta = control_input
    L = car_track_length
    #print("Forward propagate from:",state,"with steering:",delta)
    #time.sleep(1)
    little_step = step_time / 10
    for _ in range(10):
        x_new = x + v * np.cos(theta) * little_step
        y_new = y + v * np.sin(theta) * little_step
        theta_new = theta + (v / L) * np.tan(delta) * little_step
        theta_new = wrap_to_pi(theta_new)
        x, y, theta = x_new, y_new, theta_new
        if is_in_obstacle((x_new,y_new),obstacles):
            return (x_new, y_new, theta_new), False
        if not (-2 <= x_new <= 2 and -2 <= y_new <= 2):
            return (x_new, y_new, theta_new), False

    return (x, y, theta), True

def is_in_obstacle(point,obstacles):
    px, py = point

    for (ox, oy), (w, h) in obstacles:
        dx = px - ox
        dy = py - oy
        if abs(dx) <= w / 2 and abs(dy) <= h / 2:
            return True
    return False

def build_EST(start,goal,X,obstacles,radius_density, control_limits, step_time,L,trial = 0,plot = False):
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

        edges = []
        
        solution_length = {start: 0}

        iterations = 0
        plt.pause(0.1)
        x_new = start
        w = {start:0}
        discrete_steerings = [-0.5, 0, 0.5]
        free_of_obstacles = True
        while check_goal_zone(nodes,goal):
            v_src,iterations,w = sample_random_node(nodes,x_new,radius_density,iterations,w,free_of_obstacles)
            print("iteration:",iterations)  
            if iterations % 1000 == 0:
                plt.pause(5)
            control_input = np.random.choice(discrete_steerings)
            x_new,free_of_obstacles = forward_propagate(v_src, control_input, step_time,L,obstacles)
            if free_of_obstacles:
                nodes.append(x_new)
                w [x_new]= 0 
                edges.append((v_src, x_new))
                solution_length[x_new] = solution_length[v_src] + step_time * 0.2
                
                if plot:
                    plt.plot([v_src[0], x_new[0]], [v_src[1], x_new[1]], color='black')  
                    plt.plot(x_new[0], x_new[1], 'go', markersize=2)  
                    plt.quiver(x_new[0], x_new[1], 0.03*np.cos(x_new[2]),
                                0.03*np.sin(x_new[2]), angles='xy', scale_units='xy', scale=0.3, width=0.003)
                    #plt.pause(0.1)
        
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
    start = (-0.75, 0, np.deg2rad(0))
    goal = (0.75, 0, np.deg2rad(90))
    control_limits = [(0,1), (-0.5,0,0.5)] # speed, steering angle
    radius_density = 0.2
    step_time = 1.5
    trials_number = 100

    X = [xlim,ylim,thetalim]

    #gap_width = 10
    obstacles = [((0,0),(0.22,1.5))]
    
    iterations_list = np.zeros(trials_number,dtype=float)
    vertices_list = np.zeros(trials_number,dtype=float)
    solution_length_list = np.zeros(trials_number,dtype=float)
    
    L= 0.256


    iterations, vertices, solution_length, = build_EST(start,goal,X,obstacles,radius_density,
                                                        control_limits,step_time, L,trial = 0,plot = True)

if __name__ == "__main__":
    main()