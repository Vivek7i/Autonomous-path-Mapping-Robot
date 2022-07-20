import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import heapq
import cv2.aruco as aruco

# Global variables

no_of_rows = 12
no_of_cols = 12
elements = no_of_rows*no_of_cols
X = 53.6
Y = 65
graph = {}
square_width = 0
hospital = {}
patient  = []
start = [no_of_rows-1,no_of_cols-1]
trig = np.zeros(elements)
weight = np.zeros(elements)
parent = np.zeros(elements)
distances =None
direction = [[0,1] ,[0,-1] ,[-1,0] ,[1,0]]

#functions
def check_blue(R,G,B):
    if(B>=215 and G<=10 and R<=10):
        return True
    return False
def check_green(R,G,B):
#     if (abs(B_m)<=40 and abs(G_m - 227)<=40 and abs(R_m)<=40):
    if(B<=40 and G>=200 and R<=40):
        return True
    return False
def check_red(R,G,B):
#     elif (abs(B_m)<=40 and abs(G_m)<=40 and abs(R_m - 144)<=40): 
    if(B<=30 and G<=30 and R>=120):
        return True
    return False
def check_yellow(R,G,B):
#     elif (abs(B_m)<=40 and abs(G_m - 227)<=40 and abs(R_m - 227)<=40):
    if(B<=40 and G>=200 and R>=200):
        return True
    return False
def check_L_blue(R,G,B):
#     elif (abs(B_m - 230)<=40 and abs(G_m - 225)<=40 and abs(R_m - 2)<=40): 
    if(B>=200 and G>=190 and R<=30):
        return True
    return False
def check_white(R,G,B):
#     elif (abs(B_m - 227)<=40 and abs(G_m - 227)<=40 and abs(R_m - 227)<=40):
    if(B>=200 and G>=200 and R>=200):
        return True
    return False
def check_pink(R,G,B):
    if(abs(B - 211)<=40 and abs(G - 114)<=40 and abs(R - 211)<=40):
#     if(B>=215 and G<=10 and R<=10):
        return True
    return False


def add_all_edges(w,i,j):
    global no_of_rows
    if(i>0):
        add_edge(no_of_rows*(i-1)+j,no_of_rows*i+j, w)
    if(j>0):
        add_edge(no_of_rows*(i)+j-1,no_of_rows*i+j, w)
    if(i<no_of_cols-1):
        add_edge(no_of_rows*(i+1)+j,no_of_rows*i+j, w)
    if(j<no_of_rows-1):
        add_edge(no_of_rows*(i)+j+1,no_of_rows*i+j, w)


def contruct_graph(img):
    global square_width, hospital, patient, start, trig, weight, element, no_of_rows, no_of_cols,X,Y

    for i in range(elements):
        add_vertex(i)
    print(img.shape,"**")
    for i in range(no_of_cols):#col
        for j in range(no_of_rows):#row
            
            temp_img = img[(int)(j*X +45):(int)((j)*X+86), (int)(i*X +45):(int)((i)*X+86)]
            temp_img = cv2.resize(temp_img,(500,500))
            # cv2.imshow("temp_img",temp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cropping to required dimensions
            mid = (temp_img.shape[0])//2


            #finding color in the mid of cropped image 
            B_m =  temp_img[mid][mid][0]
            G_m =  temp_img[mid][mid][1]
            R_m =  temp_img[mid][mid][2]

            if(R_m<=40 and B_m<=40 and G_m<=40 ):
                continue
            
            if check_blue(R_m,G_m,B_m):
                # blue region in the centre
                lower_blue = np.array([215,0,0])
                upper_blue = np.array([235,50,10])  
                masked_blue = cv2.inRange(temp_img , lower_blue, upper_blue)

                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(255 - masked_blue,kernel,iterations = 1)

                cnt_blue,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for k in cnt_blue:   
                    area = cv2.contourArea(k)
                    if(area<0.95*temp_img.shape[0]*temp_img.shape[1] and area >=500) :
                        perimeter = cv2.arcLength(k,True) #it gives the perimeter of each shape(true is for closed shape)
                        approxCnt = cv2.approxPolyDP(k,0.02*perimeter,True) #this will give coridinates of all the corner points
                        No_of_points = len(approxCnt)
#                             print(approxCnt)
                        print(No_of_points)
                        if(No_of_points <= 3):
                            print("Triangle detected at ",(i,j))
                            for p in approxCnt:
                                if(abs((temp_img.shape[0]/2)-p[0][0])<=50):
                                    # triangle pointing up
                                    if(p[0][1]<temp_img.shape[1]/2):
                                        trig[no_of_rows*i+j]=1
                                    # down
                                    else:
                                        trig[no_of_rows*i+j]=3
                                elif(abs((temp_img.shape[1]/2)-p[0][1])<=50):
                                    # triangle pointing left
                                    if(p[0][0]<temp_img.shape[0]/2):
                                        trig[no_of_rows*i+j]=4
                                    # right
                                    else:
                                        trig[no_of_rows*i+j]=2

                        elif(No_of_points == 4):
                            print("Square detected at ",(i,j))
                            hospital["Square"] = [i,j]
                        else:
                            print("Circle detected at ",(i,j))
                            hospital["Circle"] = [i,j]
                        cv2.drawContours(temp_img, approxCnt, -1, (0,255,0),5)
                        # cv2.imshow("temp_img",temp_img)
                        # cv2.waitKey(0) 
                        # cv2.destroyAllWindows() 
                corner1 = (temp_img.shape[0])-10
                corner2 = (temp_img.shape[1])-10
                B_corner =  temp_img[corner1][corner2][0]
                G_corner =  temp_img[corner1][corner2][1]
                R_corner =  temp_img[corner1][corner2][2]
                if check_green(R_corner, G_corner, B_corner):
                    #GREEN
                    w=2
                    weight[no_of_rows*i+j] = w
                    add_all_edges(w,i,j)
                elif check_red(R_corner, G_corner, B_corner):
                    #RED
                    w=4
                    weight[no_of_rows*i+j] = w
                    add_all_edges(w,i,j) 
                elif check_yellow(R_corner, G_corner, B_corner):
                    #YELLOW
                    w=3
                    weight[no_of_rows*i+j] = w
                    add_all_edges(w,i,j)
                elif check_L_blue(R_corner, G_corner, B_corner):
                    #LIGHT BLUE
                    w=100000
                    weight[no_of_rows*i+j] = w
                    add_all_edges(w,i,j)
                elif check_white(R_corner, G_corner, B_corner):
                    #WHITE
                    w=1
                    weight[no_of_rows*i+j] = w
                    add_all_edges(w,i,j)

            elif check_pink(R_m, G_m, B_m):
                w=1000000
                weight[no_of_rows*i+j] = w
                patient.append((i,j))
                #PINK
            elif check_green(R_m, G_m, B_m): 
                #GREEN
                w=2
                weight[no_of_rows*i+j] = w
                add_all_edges(w,i,j)
            elif check_red(R_m, G_m, B_m): 
                #RED
                w=4
                weight[no_of_rows*i+j] = w
                add_all_edges(w,i,j)
            elif check_yellow(R_m, G_m, B_m): 
                #YELLOW
                w=3
                weight[no_of_rows*i+j] = w
                add_all_edges(w,i,j)
            elif check_white(R_m, G_m, B_m): 
                #WHITE
                w=1
                weight[no_of_rows*i+j] = w
                add_all_edges(w,i,j)
    # print(trig,"=trig")
    for i in range(elements) :
        if(trig[i]==1):
            if(weight[i]!=0 and check(i-1)):
                graph[i-1].remove([i,weight[i]])
            if(weight[i + no_of_rows]!=0 and check(i + no_of_rows)):
                graph[i].remove([i + no_of_rows,weight[i + no_of_rows]])
            if(weight[i - no_of_rows]!=0 and check(i - no_of_rows)):
                graph[i].remove([i - no_of_rows,weight[i - no_of_rows]])
            if(weight[i+1]!=0 and  check(i +1)):
                graph[i].remove([i + 1 ,weight[i + 1]])  
        if(trig[i]==2):
            if(weight[i]!=0 and check(i + no_of_rows)):
                graph[i+no_of_rows].remove([i,weight[i]])
            if(weight[i -1]!=0 and check(i -1)):
                graph[i].remove([i -1,weight[i -1]])
            if(weight[i - no_of_rows]!=0 and check(i - no_of_rows)):
                graph[i].remove([i - no_of_rows,weight[i - no_of_rows]])
            if(weight[i+1]!=0 and check(i + 1)):
                graph[i].remove([i + 1 ,weight[i + 1]])  
        if(trig[i]==3):
            if(weight[i]!=0 and check(i + 1)):
                graph[i+1].remove([i,weight[i]])
            if(weight[i + no_of_rows]!=0 and check(i + no_of_rows)):
                graph[i].remove([i + no_of_rows,weight[i + no_of_rows]])
            if(weight[i - no_of_rows]!=0 and check(i - no_of_rows)):
                graph[i].remove([i - no_of_rows,weight[i - no_of_rows]])
            if(weight[i-1]!=0 and check(i -1)):
                graph[i].remove([i - 1 ,weight[i - 1]]) 
        if(trig[i]==4):
            if(weight[i]!=0 and check(i - no_of_rows)):
                graph[i-no_of_rows].remove([i,weight[i]])
            if(weight[i -1]!=0 and check(i -1)):
                graph[i].remove([i -1,weight[i -1]])
            if(weight[i + no_of_rows]!=0 and check(i + no_of_rows)):
                graph[i].remove([i + no_of_rows,weight[i + no_of_rows]])
            if(weight[i+1]!=0 and check(i + 1)):
                graph[i].remove([i + 1 ,weight[i + 1]])  
    # print(print_graph(elements))

def check( i ):
    global no_of_rows
    x = i%no_of_rows
    y = i//no_of_rows
    print("x=",x)
    print("y=",y)
    if(x>=0 and x<no_of_rows and y>=0 and y<no_of_rows):
        return True
    return False

# Add a vertex to the dictionary
vertices_no = 0
def add_vertex(v):
    global graph
    global vertices_no
    if v in graph:
        print("Vertex ", v, " already exists.")
    else:
        vertices_no = vertices_no + 1
    graph[v] = []

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global graph
    # Check if vertex v1 is a valid vertex
    if v1 not in graph:
        print("Vertex ", v1, " does not exist.")
    # Check if vertex v2 is a valid vertex
    elif v2 not in graph:
        print("Vertex ", v2, " does not exist.")
    else:
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
        temp = [v2, e]
        graph[v1].append(temp)

# Print the graph
def print_graph(elements):
    global graph
    q= 0
    occur = np.zeros(elements)
    for vertex in graph:
        for edges in graph[vertex]:
            print(vertex, " -> ", edges[0], " edge weight: ", edges[1])
            occur[edges[0]] =  occur[edges[0]] + 1 
            q = q+1
    return occur,q


def calculate_distances(starting_vertex):
    global graph
    distances = {vertex: float('infinity') for vertex in graph}
    distances[starting_vertex] = 0

    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        for neighbor_info in graph[current_vertex]:
            [neighbor, weight] = neighbor_info
            distance = current_distance + weight

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances

def remove_gray_part():
    img = env.camera_feed()
    lower_mask = np.array([203+1,177+1,177+1])
    upper_mask = np.array([205-1,179-1,179-1])
    masked_img = cv2.inRange(img , lower_mask, upper_mask)
    contours,hierarchy = cv2.findContours(masked_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:   
        area = cv2.contourArea(i)
        if(area<0.9*img.shape[0]*img.shape[1]) :
            perimeter = cv2.arcLength(i,True) #it gives the perimeter of each shape(true is for closed shape)
            approxCnt = cv2.approxPolyDP(i,0.01*perimeter,True) #this will give coridinates of all the corner points
    X_min = 10000
    X_max = 0
    Y_min = 10000
    Y_max = 0 
    for i in approxCnt:
        if(X_min + Y_min>= i[0][0]+ i[0][1]):
            X_min = min(X_min, i[0][0])
            Y_min = min(Y_min, i[0][1])
        if(X_max + Y_max<= i[0][0]+ i[0][1]):
            X_max = max(X_max, i[0][0])
            Y_max = max(Y_max, i[0][1])
    new_img = img[X_min:X_max, Y_min:Y_max]
    # reduce by 10px max and min
    new_img = new_img[0+10:new_img.shape[0]-10, 0+10:new_img.shape[1]-10]
    # cv2.imshow("new_img",new_img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    # cv2.imwrite("new_arena.jpg",new_img)
    return new_img

def detect_aruco():
    # Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=2,
            markersY=2,
            markerLength=0.09,
            markerSeparation=0.01,
            dictionary=ARUCO_DICT)

    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None


    img_temp = env.camera_feed()

    gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

        #Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Make sure all 5 markers were detected before printing them out
    # if ids is not None:
    #             # Print corners and ids to the console
    #     for i, corner in zip(ids, corners):
    #         print('ID: {}; Corners: {}'.format(i, corner))

    return ids,corners
      

def target_neighbor(target):
    print(target)
    #finds the best neighbor of target where our bot can approach(best mean minimum distance)
    global direction, no_of_cols, no_of_cols, distances
    neighbor = None
    min_dist = 1e5
    (x,y) = target
    for j in direction:
        x_curr = x + j[0]
        y_curr = y + j[1]
        if(x_curr<no_of_cols and x_curr>=0 and y_curr<no_of_rows and y_curr>=0):
            if(min_dist > distances[no_of_rows*x_curr + y_curr]):
                min_dist = distances[no_of_rows*x_curr + y_curr]
                neighbor = [x_curr,y_curr]
    print(neighbor)
    return neighbor, min_dist

def retrieve_path(target_patient_neighbor,min_dist,start):
    global direction,no_of_rows,no_of_cols,graph,weight,distances
    curr = target_patient_neighbor[0]*no_of_cols+target_patient_neighbor[1] 
    path = []
    while(curr!=(no_of_cols*start[0]+start[1])):
        # print("curr = ",curr)
        path.append([curr//no_of_cols,curr%no_of_cols])
        curr =(int)(parent[curr])
    path.append(start)
    path.reverse()
    return path

def get_optimised_path(org_path):
    path= org_path.copy()
    x=len(path)
    final_path=[]
    for i in range(1,x-1):
        if(path[i][0]==path[i+1][0] and path[i-1][0]==path[i][0]  or path[i][1]==path[i+1][1] and path[i-1][1]==path[i][1]):
            continue
        else:
            final_path.append([path[i][0],path[i][1]])
    final_path.append([path[x-1][0],path[x-1][1]])
    return final_path    


def traversal(start, final_target):
    #start final_path final_target = Pt
    global distances
    logic = 1
    print("target = ",  final_target)
    distances = calculate_distances(no_of_rows*start[0]+start[1])
    print(distances)            
    weight[no_of_rows*start[0]+start[1]] = 100000
    neighbor, min_dist = target_neighbor(final_target)
    print("neighbor=",neighbor,",min_dist = ", min_dist )
    path = retrieve_path(neighbor,min_dist,start)
    print("path",path)
    final_path = get_optimised_path(path)
    print(final_path)
    prev_set_point = start

    ids, corners = detect_aruco()
    bot_vector = corners[0][0][0] - corners[0][0][3]
    bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
    path_vector = [final_path[0][0] - start[0],final_path[0][1] - start[1]]
    cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
    dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]
    print(corners)
    print("bot_vector",bot_vector)
    print("path_vector",path_vector)
    print("cross_product",cross_product)
    print("dot_product",dot_product)
    if(abs(cross_product)<=4 and dot_product<=0):
        print("logic=-1")
        logic = -1
    flag = -1
    for set_point in final_path:
        if(set_point[0]==prev_set_point[0]):
            flag = 0
        else:
            flag = 1
        print("set_point=",set_point)
        ids, corners = detect_aruco()
        bot_vector = logic*(corners[0][0][0] - corners[0][0][3])
        bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
        path_vector = [set_point[0] - bot_pos[0],set_point[1] - bot_pos[1]]
        cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
        dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]
        last = []
        b=-1
        while(1):
            b=b+1
            if( b%10!=0):
                p.stepSimulation()
                env.move_husky(last[0],last[1],last[2],last[3])
                continue
            ids, corners = detect_aruco()
            if(len(corners) == 0):
                p.stepSimulation()
                env.move_husky(last[0],last[1],last[2],last[3])
                continue
            bot_vector = logic*(corners[0][0][0] - corners[0][0][3])
            bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
            path_vector = [set_point[0] - prev_set_point[0],set_point[1] - prev_set_point[1]]
            cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
            dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]
            # print("bot_vector",bot_vector)
            # print("path_vector",path_vector)
            # print("cross_product",cross_product)
            # print("dot_product",dot_product)
            if(cross_product<0):
                # print("rotate left")
                if(logic == 1):
                    print("positive logic ka left")
                    p.stepSimulation()
                    env.move_husky(-4, 8, -2, 8)
                    # last = [-4, 8, -4, 8]
                    last = [-4, 8, -2, 8]
                if(logic == -1):
                    print("negative logic ka left")
                    p.stepSimulation()
                    env.move_husky(-4, 2, -8, 4)
                    # last = [8, -4, 8, -4]
                    last = [-4, 2, -8, 4]

            elif(cross_product>0):
                # print("rotate right")
                if(logic == 1):
                    print("positive logic ka right")
                    p.stepSimulation()
                    env.move_husky(8, -4, 8, -2)
                    # last = [8, -4, 8, -4]
                    last = [8, -4, 8, -2]
                if(logic == -1):
                    print("negative logic ka right")
                    p.stepSimulation()
                    env.move_husky(2, -4, 4, -8)
                    # last = [8, -4, 8, -4]
                    last = [2, -4, 4, -8]    
            # print(cross_product,"**",dot_product)
            if(abs(cross_product)<=2):
                print("BREAK**")
                break
        
        ids, corners = detect_aruco()
        bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
        # print(bot_pos)
        # print((set_point[0]*X+Y), (set_point[1]*X+Y),"*")
        if(flag==0):
            dist_bot_and_set_point = (bot_pos[1]-(set_point[1]*X+Y))**2
        else:
            dist_bot_and_set_point = (bot_pos[0]-(set_point[0]*X+Y))**2
        # print(dist_bot_and_set_point)
        b= -1
        last_dist = dist_bot_and_set_point
        wrong = 0
        if(logic==-1):
            thresh = 1000
            if( (set_point[0]-5.5)**2 + (set_point[1]-5.5)**2  > 25 ):
                thresh = 500
        else:
            thresh = 50
            if( (set_point[0]-5.5)**2 + (set_point[1]-5.5)**2  > 25 ):
                thresh = 10
        while(dist_bot_and_set_point>=thresh and wrong<=2):
            b=b+1            
            if(b%10== 0):
                ids, corners = detect_aruco()
                if(len(corners)==0):
                    continue
                bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
                if(last_dist < dist_bot_and_set_point):
                    wrong+=1
                else:
                    wrong = 0
                last_dist = dist_bot_and_set_point

                # dist_bot_and_set_point = (bot_pos[0]-(set_point[0]*X+Y))**2 + (bot_pos[1]-(set_point[1]*X+Y))**2
                if(flag==0):
                    dist_bot_and_set_point = (bot_pos[1]-(set_point[1]*X+Y))**2
                else:
                    dist_bot_and_set_point = (bot_pos[0]-(set_point[0]*X+Y))**2
                print(dist_bot_and_set_point,"**")
            p.stepSimulation()
            if(logic==1):
                env.move_husky(4, 4, 4, 4)
            else:
                env.move_husky(-4, -4, -4, -4)

            
        prev_set_point = set_point
        start_time = time.time()
        seconds = 0.5

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            p.stepSimulation()
            env.move_husky(0, 0, 0, 0)  
            if elapsed_time > seconds:
                # print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
                break
    path_vector = [final_target[0] - prev_set_point[0],final_target[1] - prev_set_point[1]]
    return logic


def PINK_TRAVERSAL(curr_pos,target_node,logic):
    
    ids, corners = detect_aruco()
    bot_vector = logic*(corners[0][0][0] - corners[0][0][3])
    bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
    path_vector = [target_node[0] - curr_pos[0],target_node[1] - curr_pos[1]]
    cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
    dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]
    last = []
    b=-1
    while(1):
        b=b+1
        if( b%10!=0):
            p.stepSimulation()
            env.move_husky(last[0],last[1],last[2],last[3])
            continue
        ids, corners = detect_aruco()
        if(len(corners) == 0):
            p.stepSimulation()
            env.move_husky(last[0],last[1],last[2],last[3])
            continue
        bot_vector = logic*(corners[0][0][0] - corners[0][0][3])
        bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
        cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
        dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]

        if(cross_product<0):
            # print("rotate left")
            if(logic == 1):
                print("positive logic ka left")
                p.stepSimulation()
                env.move_husky(-4, 8, -2, 8)
                # last = [-4, 8, -4, 8]
                last = [-4, 8, -2, 8]
            if(logic == -1):
                print("negative logic ka left")
                p.stepSimulation()
                env.move_husky(-4, 2, -8, 4)
                # last = [8, -4, 8, -4]
                last = [-4, 2, -8, 4]

        elif(cross_product>0):
            # print("rotate right")
            if(logic == 1):
                print("positive logic ka right")
                p.stepSimulation()
                env.move_husky(8, -4, 8, -2)
                # last = [8, -4, 8, -4]
                last = [8, -4, 8, -2]
            if(logic == -1):
                print("negative logic ka right")
                p.stepSimulation()
                env.move_husky(2, -4, 4, -8)
                # last = [8, -4, 8, -4]
                last = [2, -4, 4, -8]    
        # print(cross_product,"**",dot_product)
        if(abs(cross_product)<=2):
            print("BREAK**")
            break
    
    ids, corners = detect_aruco()
    bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
    dist_bot_and_NT = (bot_pos[0]-(int)(target_node[0]*X+Y))**2 + (bot_pos[1]-(int)(target_node[1]*X+Y))**2
    # print(dist_bot_and_Pt)
    b=0
    last_dist = dist_bot_and_NT
    wrong = 0
    if(logic == -1):
        thresh = 20
    else:
        thresh = 20
    while(dist_bot_and_NT>=thresh and wrong<=2):
        if(b%10 == 0):
            ids, corners = detect_aruco()
            bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
            if(last_dist<dist_bot_and_NT):
                wrong+=1
            else:
                wrong = 0
            last_dist =dist_bot_and_NT
            dist_bot_and_NT = (bot_pos[0]-(int)(target_node[0]*X+Y))**2 + (bot_pos[1]-(int)(target_node[1]*X+Y))**2
            print(dist_bot_and_NT)
        p.stepSimulation()
        if(logic==1):
            env.move_husky(4, 4, 4, 4)
        else:
            env.move_husky(-4, -4, -4, -4)
        b=b+1
    time.sleep(2)
        
def small_traversal(logic,target_node):
    ids, corners = detect_aruco()
    bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
    dist_bot_and_NT = (bot_pos[0]-(int)(target_node[0]*X+Y))**2 + (bot_pos[1]-(int)(target_node[1]*X+Y))**2
    b=0
    last_dist = dist_bot_and_NT
    wrong = 0
    if(logic == -1):
        thresh = 50
    else:
        thresh = 50
    while(dist_bot_and_NT>=X**2 and wrong<=2):
        if(b%10 == 0):
            ids, corners = detect_aruco()
            bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
            if(last_dist<dist_bot_and_NT):
                wrong+=1
            else:
                wrong = 0
            last_dist =dist_bot_and_NT
            dist_bot_and_NT = (bot_pos[0]-(int)(target_node[0]*X+Y))**2 + (bot_pos[1]-(int)(target_node[1]*X+Y))**2
            print(dist_bot_and_NT)
        p.stepSimulation()
        if(logic==1):
            env.move_husky(4, 4, 4, 4)
        else:
            env.move_husky(-4, -4, -4, -4)
        b=b+1


def reverse_small_traversal(logic,neighbor,target_node):
    ids, corners = detect_aruco()
    bot_vector = (corners[0][0][0] - corners[0][0][3])
    path_vector = [target_node[0] - neighbor[0],target_node[1] - neighbor[1]]
    cross_product = bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0]
    bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
    dist_bot_and_NT = (bot_pos[0]-(int)(neighbor[0]*X+Y))**2 + (bot_pos[1]-(int)(neighbor[1]*X+Y))**2
    print(cross_product)
    if(abs(cross_product)<6):
        return
    b=0
    while(dist_bot_and_NT<(1000)):
        if(b%10 == 0):
            ids, corners = detect_aruco()
            bot_pos = (corners[0][0][0] + corners[0][0][3] + corners[0][0][1] + corners[0][0][2])/4
            dist_bot_and_NT = (bot_pos[0]-(int)(neighbor[0]*X+Y))**2 + (bot_pos[1]-(int)(neighbor[1]*X+Y))**2
            print(dist_bot_and_NT)
        p.stepSimulation()
        if(logic==1):
            env.move_husky(-4, -4, -4, -4)
        else:
            env.move_husky(4, 4, 4, 4)
        b=b+1
    

if __name__=="__main__":
    env = gym.make("pix_main_arena-v0")
    env.remove_car()
    img = env.camera_feed()
    env.respawn_car()
    square_width = (int)(img.shape[0]/no_of_rows)

#new
    contruct_graph(img)


    # detect_aruco()
    
    while True:
        p.stepSimulation()
        start = [11,11]
        # time.sleep(15)
        patient.reverse()
        for Pt in patient:
            logic = traversal(start, Pt)
            neighbor, min_dist = target_neighbor(Pt)
            if(logic==-1):
                small_traversal(logic,Pt)
            env.remove_cover_plate(Pt[1],Pt[0])
            if(logic==-1):
                print("reverse = ", logic)
                reverse_small_traversal(logic,neighbor,Pt)
            
            next_target = None
            #Box identification Part
#### PINK PART IDENTIFICATION
            pink_img = env.camera_feed()

            pink_img = pink_img[(int)(X*Pt[1] + 45): (int)(X*Pt[1] + 85), (int)(X*Pt[0] + 45): (int)(X*Pt[0] + 85)]
            
            lower_blue = np.array([215,0,0])
            upper_blue = np.array([235,50,10])
            masked_blue = cv2.inRange(pink_img , lower_blue, upper_blue)

            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(255 - masked_blue,kernel,iterations = 1)

            cnt_blue,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for k in cnt_blue:   
                area = cv2.contourArea(k)
                if(area<0.95*pink_img.shape[0]*pink_img.shape[1]) :
                    perimeter = cv2.arcLength(k,True) #it gives the perimeter of each shape(true is for closed shape)
                    approxCnt = cv2.approxPolyDP(k,0.02*perimeter,True) #this will give coridinates of all the corner points
                    No_of_points = len(approxCnt)
                    print(No_of_points,"**")
                    if(No_of_points == 4):
                        print("Square detected at ",hospital["Square"])
                        next_target = hospital["Square"]
                    else:
                        print("Circle detected at ",hospital["Circle"])
                        next_target = hospital["Circle"]
#### PINK PART IDENTIFICATION
            curr_pos, min_dist = target_neighbor(Pt)
            PINK_TRAVERSAL(curr_pos,Pt,logic)

            logic = traversal(Pt, next_target)
            curr_pos, min_dist = target_neighbor(next_target)
            PINK_TRAVERSAL(curr_pos,next_target,logic)

            start = next_target

        break

    # time.sleep(100)
# thresh , r = i2+j2

# thresh = 100  - r* k 