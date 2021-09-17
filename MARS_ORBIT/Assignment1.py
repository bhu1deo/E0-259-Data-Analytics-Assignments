import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import time
from datetime import date
from numpy import *

def compute_time(Y,M,D,H,minutes):
    d_ref = date(1580,11,18)                 # reference date here :: 
    d_1 = date(Y,M,D) 
    delta = d_1 - d_ref
    time = delta.days + H/24 + minutes/(24*60) - 1/24 - 31/(24*60)        
    return time 

def compute_degrees(ZI,deg,minute,sec):
    degree = ZI*30 + deg + minute/60 + sec/3600
    return degree


def create_oppositions_array(df):
    # create array :: containing time and degree info here :: 

    df['Degree_abs'],df['Days_abs'] = df.apply(lambda df : compute_degrees(df['ZodiacIndex'],df['Degree'],df['Minute.1'],df['Second']), axis = 1),df.apply(lambda df : compute_time(df['Year'],df['Month'],df['Day'],df['Hour'],df['Minute']), axis = 1)
    df_req = df[['Degree_abs','Days_abs']]
    oppositions = df_req.to_numpy()
    oppositions[:, [1, 0]] = oppositions[:, [0, 1]]
    return oppositions

def gen_points(centre,angles):            # Given angles and the centre get the points on those lines here 
    # generate 2 set of points for each line here :: 
    r = 15 
    v = centre
    points = np.array([np.array([v[0],v[1],v[0]+r*np.cos(np.radians(ang)),v[1]+r*np.sin(np.radians(ang))]) for ang in angles])
    return points



def angle_between_lines(pt1,pt2): # Both Lines Emanate from the SUN here 
    slope_1 = pt1[1]/pt1[0]
    slope_2 = pt2[1]/pt2[0]
    angle = np.arctan(abs((slope_1-slope_2)/(1+slope_1*slope_2)))
    angle = np.degrees(angle)
    return angle


def intersection_line_circle(line_start,line_deg,circle_rad,circle_centre):
    line_rad = np.radians(line_deg)           # Value already provided in radians here 
    diff = line_start-circle_centre
    coeff = [1,2*(np.cos(line_rad)*(diff[0])+np.sin(line_rad)*(diff[1])),np.linalg.norm(diff)**2-circle_rad**2]
    dist = max(np.roots(coeff))                # We need the positive distance here and of course equant lies inside the circle 
    return line_start+np.array([dist*np.cos(line_rad),dist*np.sin(line_rad)])

def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def find_projected_points(equant_coor,equant_ref,oppositions,circle_centre,circle_radius,s):             # For every opposition there will be 2 projected points :: one from the line emanating from the equant and the other from the line emanating from the SUN 
    # long and equant angles here :: long_angle -->> oppositions data 
    # Equant Angle :: From Equant Reference and time period between oppositions data 
    long_angles = oppositions[:,1]         # In degrees 
    time_days = oppositions[:,0]
    time_diff = np.ediff1d(time_days)               # Length is 11 here 
    equant_angles = np.array([equant_ref])           # In degrees 
    for i in range(11):
        curr = equant_angles[i]            # current angle here 
        new = curr + time_diff[i]*s        # s is in deg/days
        equant_angles = np.append(equant_angles,[new%360])
        
    long_points = gen_points(np.array([0,0]),long_angles)
    equant_points = gen_points(equant_coor,equant_angles)        # Points have been generated for the longitudes and the dotted lines here 


    for i in range(12):
        p11 = long_points[i,0:2]
        p21 = long_points[i,2:4]
        p31 = equant_points[i,0:2]
        p41 = equant_points[i,2:4]
        
        p1 = intersection_line_circle(equant_coor,equant_angles[i],circle_radius,circle_centre)      # the Angle passed is in degrees here 
        p2 = intersection_line_circle(np.array([0,0]),long_angles[i],circle_radius,circle_centre)
        if(i==0):
            coordinates = np.array([seg_intersect(p11,p21,p31,p41)])
            intersections = np.array([p1,p2])
        else:
            coordinates = np.append(coordinates,np.array([seg_intersect(p11,p21,p31,p41)]),axis=0)
            intersections = np.append(intersections,np.array([p1,p2]),axis=0)
    
    return intersections,coordinates,long_points,equant_points

def MarsEquantModel(c,r,e1,e2,z,s,oppositions):
    # angular error for each opposition and maximum angular error here ::
    """
    Following units here :: SUN at the centre of the coordinate frame 
    
    c :: Angle in degrees distance 1 from the sun :: Sun fixed at the origin 
    r :: relative value to c's distance from circle's centre 
    e1 :: distance of equant from the SUN relative to sun-centre distance 
    e2 :: deg. rel. to ref. longitude so from the horizontal axis the angle is (e2+z) here 
    z :: deg. w.r.t SUN-Aries line here 
    s :: deg. per day here speed of mars 
    oppositions :: numpy array here 
    
    """
    circle_centre = np.array([np.cos(np.radians(c)),np.sin(np.radians(c))])

    equant_centre = np.array([e1*np.cos(np.radians(e2+z)),e1*np.sin(np.radians(e2+z))])
    
    intersections,_,_,_ = find_projected_points(equant_centre,z,oppositions,circle_centre,r,s)
    
#     print(long_points)

    # Here the intersections are stored in pairs of 2 :: the dotted line and the solid line with the circle here :: 
    for i in range(12):
        pt1 = intersections[2*i,:]
        pt2 = intersections[2*i+1,:]          # The intersections need to be connected with the SUN and then the angle between the lines needs to be found out here
        angle = angle_between_lines(pt1,pt2)
        if(i==0):
            errors = np.array([angle])
        else:
            errors = np.append(errors,np.array([angle]),axis=0)
    
    # Convert error to minutes here 
    return 60*errors,np.max(60*errors)

def bestOrbitInnerParams(r,s,oppositions):
    """
    Fix r and s. Do a discretised exhaustive search over c, over e = (e1,e2), 
    and over z to minimise the maximum angular error for the given r and s. 
    Your outputs should be the best parameters, the angular error for each opposition, and the maximum angular error, as follows.
    c,e1,e2,z,errors,maxError = bestOrbitInnerParams(r,s,oppositions). 

    Please note that this Exhaustive search would take a lot of time and hence all the optimal parameters are found using 
    neighbourhood search techniques after a crude rough cut aproximation. 
 
    """

    # We want to minimise the max error over all possible oppositions here 
    minimum = 100
    
    
    for c in np.linspace(1,360,36000):               # precision 0.01 degree henceforth                
        for e1 in np.linspace(1,5,400):              # precision 0.01 units   
            for e2 in np.linspace(1,360,36000):
                for z in np.linspace(1,360,36000):
                    errors,maxError = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
                    if(minimum>maxError):
                        minimum = maxError
                        errors_opt = errors
                        maxError_opt = maxError
                        c_opt = c
                        e1_opt = e1
                        e2_opt = e2
                        z_opt = z
                        
    
    return c_opt,e1_opt,e2_opt,z_opt,errors_opt,maxError_opt

def bestS(r,oppositions):
    """
    Fix r. Do a discretised search for s (in the neighbourhood of 360 degrees over 687 days; for each s, you will use the function developed in question 2). 
    Your outputs should be the best s, the angular error for each opposition, and the maximum angular error, as follows.
    s,errors,maxError = bestS(r,oppositions)
    """
    
    # Vary T and then get the optimum value of T and then return the optimum value of s here :: 
    
    # These parameters would be obtained from the exhaustive search ::          
    z = 55.86                     
    e2 = 92.85                 
    e1 = 2.04                  
    c = 146.16 
    
    minimum = 100                   # large value initially here 
    
    for T in np.linspace(686,688,2000):             # precision of 0.01 here 
        s = 360/T
        errors,maxError = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
        if(minimum>maxError):
            minimum = maxError
            errors_opt = errors
            maxError_opt = maxError
            s_opt = s
            T_opt = T
        
    
    


    return s_opt,errors_opt,maxError_opt


def bestR(s,oppositions):
    """
    Fix s. Do a discretised search for r (in the neighbourhood of the average distance of the black dots, 
                                          which are described in slide 31, from the centre; again, for each r,
                                          you will use the function developed in question 2).
    Your outputs should be the best r, the angular error for each opposition, and the maximum angular error. (over all oppositions) 
    r,errors,maxError = bestR(s,oppositions)

    """
    z = 55.86                     
    e2 = 92.85                 
    e1 = 2.04                  
    c = 146.16 
    
    minimum = 100                   # large value initially here 
    
    for r in np.linspace(10,12,2000):             # precision of 0.01 here 
        errors,maxError = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
        if(minimum>maxError):
            minimum = maxError
            errors_opt = errors
            maxError_opt = maxError
            r_opt = r

    return r_opt,errors_opt,maxError_opt

def bestMarsOrbitParams(oppositions):
    """
    Write a wrapper that will search iteratively over r and s, starting from an initial guess.
    Your outputs should be the  parameters c, e, z, r, and s, the angular error for each opposition, 
    and the maximum angular error.
    r,s,c,e1,e2,z,errors,maxError = bestMarsOrbitParams(oppositions)
    """
    minimum = 100 
    for r in np.linspace(10.94,10.98,4):
        for T in np.linspace(686.89,686.93,4):
            for c in np.linspace(146.1,146.2,10):
                for e1 in np.linspace(1.95,2.05,10):
                    for e2 in np.linspace(92.8,92.9,10):
                        for z in np.linspace(55.8,55.9,10):
                            s = 360/T
                            errors,maxError = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
                            
                            if(minimum>maxError):
                                minimum = maxError
                                errors_opt = errors
                                maxError_opt = maxError
                                r_opt = r
                                T_opt = T
                                c_opt = c
                                e1_opt = e1
                                e2_opt = e2
                                z_opt = z
                            
                            
    s_opt = 360/T_opt                        
    return r_opt,s_opt,c_opt,e1_opt,e2_opt,z_opt,errors_opt,maxError_opt

def plot(c,r,e1,e2,z,s,oppositions):
    """
    Plot contains : Equant dotted lines :: SUN longitudes :: Equant :: SUN :: center :: Intersections of lines and projections onto the circle 
    """
    # equant Coor.
    equant_centre = np.array([e1*np.cos(np.radians(e2+z)),e1*np.sin(np.radians(e2+z))])
    # Centre coor. 
    circle_centre = np.array([np.cos(np.radians(c)),np.sin(np.radians(c))])
    # SUN 
    SUN = np.array([0,0])
    fig, ax = plt.subplots()
    plt.scatter(equant_centre[0],equant_centre[1],c='r',label='equant')
    plt.scatter(circle_centre[0],circle_centre[1],c='b',label='circle_centre')
    plt.scatter(SUN[0],SUN[1],c='k',label='SUN')
    # Equant Dotted Lines start and end points :: generate points on the lines and plot a line through them here :: 
    intersections,coor,long_points,equant_points = find_projected_points(equant_centre,z,oppositions,circle_centre,r,s)
    for i in range(12):
        plt.plot(long_points[i,0::2],long_points[i,1::2],label=i+1)
        plt.plot(equant_points[i,0::2],equant_points[i,1::2],'--')
        plt.scatter(coor[i,0],coor[i,1],c='k')
    circle = plt.Circle(circle_centre, r, color='k',fill=False)
    ax.add_patch(circle)
    # Show the Plot 
    plt.legend()

    
    plt.axis("equal")
    plt.axis([-35, 35, -35, 35])
#     plt.savefig('Final_Plot.png')
    plt.show()


def main():
    print("Please keep the data file in the same folder")
    data_file = pd.read_csv('01_data_mars_opposition_updated.csv')
    oppositions = create_oppositions_array(data_file)
    
    r = 10.96                            # relative to center to sun distance here 
    s = 360/686.91          # approximate initial start point here 
    z = 55.855                                       # equant-0 here 
    e2 = 92.9                               # equant angle w.r.t equant-0 longitude 
    e1 = 2.0388                               # equant to sun :: considering sun to centre of circle is 1 unit here 
    c = 146.3                             # The angle in degrees from the sun-aries line here 

    # Part-1 Compute Error Discrepancy :: 
    print("\nPART-1\n")
    errors,maxError = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
    print("\nPrinting errors and maxError for the above values\n")
    print(errors)
    print("\n")
    print(maxError)
    
    # Part-2 exhaustive search over c, over e = (e1,e2), and over z :: Takes a lot of time EXHAUSTIVE SEARCH
    print("\nPART-2, can be commented out as Exhaustive search takes very long time\n")
    r = 11
    s = 360/687
    
    c,e1,e2,z,errors,maxError = bestOrbitInnerParams(r,s,oppositions)
    
    print("\nPrinting Best Inner Orbit parameters \n")

    print('Printing c:{},e1:{},e2:{},z:{},errors:{},maxError:{}'.format(c,e1,e2,z,errors,maxError))
    
    # Part-3 :: Discreetised Search for S :: 
    print("\nPART-3\n")
    
    r = 11
    
    s,errors,maxError = bestS(r,oppositions)

    print('\nPrinting s:{},errors:{},maxError:{}'.format(s,errors,maxError))
    
    # Part-4 :: Discretised search for R :: use the above obtained s :: do not use a CRUDE estimate here 
    print("\nPART-4\n")

    r,errors,maxError = bestR(s,oppositions)

    print('\nPrinting r:{},errors:{},maxError:{}'.format(r,errors,maxError))
    
    # Part-5 Best Parameters for the whole problem :: 
    print("\nPART-5\n")
    
    r_opt,s_opt,c_opt,e1_opt,e2_opt,z_opt,errors_opt,maxError_opt = bestMarsOrbitParams(oppositions)

    print('\nPrinting r:{},s:{},c:{},e1:{},e2:{},z:{},errors:{},maxError:{}'.format(r_opt,s_opt,c_opt,e1_opt,e2_opt,z_opt,errors_opt,maxError_opt))

    # Plotting here :: 
    
    plot(c_opt,r_opt,e1_opt,e2_opt,z_opt,s_opt,oppositions)

    
    
    
if __name__=="__main__":
    main()