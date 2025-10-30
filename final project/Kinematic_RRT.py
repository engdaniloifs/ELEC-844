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
        while random_node is None or is_in_obstacle(random_node,obstacles):
            random_node = (np.random.uniform(*X[0]), np.random.uniform(*X[1]))
            iterations += 1
    return random_node, iterations

def find_nearest_node(nodes, point):
    return min(nodes, key=lambda node: distance(node, point))

def distance(node1, node2):
    return np.hypot(node1[0] - node2[0], node1[1] - node2[1])

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def steer(nearest, point, step_size,car_variables,obstacles,goal):
    speed = car_variables[0]
    maximum_steering = car_variables[1]
    L = car_variables[2]
    step_time = step_size/speed
    T = 0
    t = step_time/5
    x_new = nearest[0]
    y_new = nearest[1]
    theta_new = nearest[2]
    while (distance(nearest, point) > (speed*T) and (speed*T)<step_size):
        
        alpha = theta_new - np.atan2(point[1] - y_new, point[0] - x_new)
        alpha = wrap_to_pi(alpha)
    
        phi = -np.atan2(alpha*L,speed*step_time)

        phi = np.clip(phi,-maximum_steering,maximum_steering)
        
        x_new = x_new + speed*t*np.cos(theta_new)
        y_new = y_new + speed*t*np.sin(theta_new)
        theta_new = wrap_to_pi(theta_new + (speed*t*np.tan(phi))/L)
        

        if is_in_obstacle((x_new,y_new),obstacles):
            return (x_new, y_new, theta_new),step_size, False
        T += t
    distance_to_goal = distance((x_new,y_new),goal)
    if (distance_to_goal < step_size):
        x_new, y_new, theta_new = goal

    return (x_new, y_new,theta_new), step_size, True


def  build_RRT(start,goal,xlim,ylim,obstacles,epsilon,step_size,car_variables,trial,plot = False):
    # with plot on
    np.random.seed(trial)
    if plot:
        x_start,y_start, theta_start = start 
        x_goal,y_goal,theta_goal = goal
        plt.plot(x_start,y_start, 'bo', markersize=4, label="Start")
        plt.quiver(x_start, y_start, 2*np.cos(theta_start), 2*np.sin(theta_start), angles='xy', scale_units='xy', scale=1, width=0.002)
        plt.plot(x_goal,y_goal, 'ro', markersize=4, label="Goal")
        plt.quiver(x_goal, y_goal, 2*np.cos(theta_goal), 2*np.sin(theta_goal), angles='xy', scale_units='xy', scale=1, width=0.002)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title("RRT in progress")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        ax = plt.gca()
        for (ox, oy), (w, h) in obstacles:
            
            rect = patches.Rectangle((ox - w/2, oy - h/2), w, h, color='gray')
            
            ax.add_patch(rect)

        X = [xlim, ylim]
        
        nodes = [start]
        edges = []
        solution_length = {start: 0}
        iterations = 0
        while goal not in nodes:
            random_node, iterations = sample(X,epsilon,goal,obstacles,iterations)
            nearest_node = find_nearest_node(nodes,random_node)
            new_node,cost,is_edge_valid = steer(nearest_node, random_node, step_size,car_variables,obstacles,goal)
            if iterations % 50 == 0:
                plt.pause(1)
            if is_edge_valid:
                

                nodes.append(new_node)
                edges.append((nearest_node,new_node))
                
                cost = solution_length[nearest_node]+ cost
                solution_length[new_node] = cost
                
                plt.plot([nearest_node[0], new_node[0]], [nearest_node[1], new_node[1]], color='black')  
                plt.plot(new_node[0], new_node[1], 'go', markersize=2)  
                plt.quiver(new_node[0], new_node[1], 2*np.cos(new_node[2]),
                            2*np.sin(new_node[2]), angles='xy', scale_units='xy', scale=1, width=0.002)
                

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
        
        for (p, c) in reversed(path):   # go from start → goal
            x1, y1,theta1 = p
            x2, y2,theta2 = c
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)  
            
        plt.pause(1)
        plt.title(f"RRT Finished - Trial number {trial+1}")
        print("Iterations:",iterations)
        print("Vertices:",len(nodes))
        print("Solution length:", solution_length[goal])
        plt.show()
        return iterations, len(nodes), solution_length[goal], path[-2][1]
    
    #without plot on
    X = [xlim, ylim]
    
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

    xlim,ylim = (0, 100),(0,100)
    start = (25, 50, np.deg2rad(0))
    goal = (75, 50, np.deg2rad(90))
    epsilon = 0.01
    step_size = 5
    trials_number = 100

    gap_width = 25
    obstacles = [((50,50),(10,100-2*gap_width))]
    
    iterations_list = np.zeros(trials_number,dtype=float)
    vertices_list = np.zeros(trials_number,dtype=float)
    solution_length_list = np.zeros(trials_number,dtype=float)
    
    #car_variables
    maximum_steering = 0.6 #rad
    speed = 1 #
    L = 1.5 # The vehicle's track length.
    car_variables = [speed,maximum_steering, L]
    phi_circle = np.atan2(goal[1]-50,goal[0]-50)  # current angle
    v_goal = speed
    angular_velocity_goal = v_goal/25  # radius is 25
    while distance(start[:2],goal[:2]) > step_size:
        iterations, vertices, solution_length, next_position = build_RRT(start,goal,xlim,ylim,obstacles,epsilon,
                                                        step_size,car_variables,trial = 0,plot = True)
        start = (next_position[0],next_position[1],next_position[2])
        
        phi_circle = wrap_to_pi(phi_circle +  angular_velocity_goal*step_size)           # current angle
        x_new_goal = 50 + 25 * np.cos(phi_circle)
        y_new_goal = 50 + 25 * np.sin(phi_circle)
        theta_new_goal = wrap_to_pi(phi_circle + np.pi/2)
        goal = (x_new_goal,y_new_goal,theta_new_goal)
        
    
    iterations_list[0] = iterations
    vertices_list[0] = vertices
    solution_length_list[0] = solution_length

    

    for trial in range(1,trials_number):

        iterations, vertices, solution_length = build_RRT(start,goal,xlim,ylim,obstacles,epsilon,step_size,trial)
    
        iterations_list[trial] = iterations
        vertices_list[trial] = vertices
        solution_length_list[trial] = solution_length

    iterations_mean = np.mean(iterations_list)
    vertices_mean = np.mean(vertices_list)
    solution_length_mean = np.mean(solution_length_list)

    print("Iterations mean:",iterations_mean)
    print("Vertices mean:",vertices_mean)
    print("Solution length mean:", solution_length_mean)


    plt.show()

if __name__ == "__main__":
    main()