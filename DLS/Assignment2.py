# We write what we have ::

import numpy as np 
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
import pandas as pd 
from scipy.optimize import minimize


def get_data(path):        # path of the CSV file should be passed here 
    
    data_frame = pd.read_csv(path)

    df = data_frame[['Match','Innings', 'Over', 'Total.Overs','Total.Runs','Innings.Total.Runs', 'Wickets.in.Hand']]

    df = df[df['Innings'] == 1]            # 1st innings only 

    df['u'] = df['Total.Overs'] - df['Over']         # overs left  

    df['Z'] = df['Innings.Total.Runs'] - df['Total.Runs']                  # Runs scored from now on  

    df['w'] = df['Wickets.in.Hand']                  # Wickets left 

    new_df = df.drop_duplicates(subset=['Match'])                     # Unique match number rows here  

    u_50_w_10 = np.mean(new_df['Innings.Total.Runs'].unique())        # 50 overs left and 10 wickets in hand here 
    
    df = df[['u','Z','w']]                                # Rest of the columns are not required 

    df.loc[(df['Z'] < 0), 'Z'] = 0                          # No negative runs allowed 

    df.loc[(df['u'] == 0) | (df['w'] == 0), 'Z'] = 0     # if no resources then no runs can be scored here 

    data = np.zeros((51,11))             # 0-50 and 0-10 here 

    df_mean = df.groupby(['u', 'w'],as_index=False).mean()

    data[np.array(df_mean['u'].values),np.array(df_mean['w'].values)] = df_mean['Z'].values    # A data matrix 
    
    data[50,10] = u_50_w_10             # Needs to be separately computed and stored here 

    data = np.transpose(data)  
    
    return data


def func_20_params(x, Z, b):              # method-1
    return Z*(1 - np.exp(-b*x))                 # slope : Z*b at x = 0 here

def func_11_params(x, Z, L):              # method-2 
    return Z*(1 - np.exp(-L*x/Z))               # slope : Z*L/Z = L here at x = 0 


def DuckworthLewis20Params(data):       # optimize and return the params 
    # for every wicket there will be a plot here : only non-zero entries in the data array would be used for interpolation 
    Z0 = np.zeros(11)
    b  = np.zeros(11) 
    
    for w in range(1,11):                     # 1 to 10 wickets here 
        data_w = data[w,:]              # all the overs data given a particular w here 
        non_zero_indices = np.nonzero(data_w)
        data_w = data_w[non_zero_indices]         # only non-zero datapoints to be taken here
        
        def rosen(x):
            val = x[0]*(1 - np.exp(-x[1]*non_zero_indices[0]))
            return np.linalg.norm(data_w-val)**2
        
        res = minimize(rosen, np.array([w*20,0.05]), method='L-BFGS-B',
               options={'disp': False})
        
        Z0[w],b[w] = res.x
    return Z0,b                     # A list of 10 Z0 params and 10 b values here


def DuckworthLewis11Params(data):            # optimize and return the params 
    # for every wicket there will be a plot here : only non-zero entries in the data array would be used for interpolation 
    Z0 = np.zeros(11)
    L  = np.zeros(11)
    for w in range(1,11):                     # 1 to 10 wickets here 
        data_w = data[w,:]              # all the overs data given a particular w here 
        non_zero_indices = np.nonzero(data_w)
        data_w = data_w[non_zero_indices]         # only non-zero datapoints to be taken here
        
        def rosen(x):
            val = x[0]*(1 - np.exp(-x[1]*non_zero_indices[0]/x[0]))
            return np.linalg.norm(data_w-val)**2
        
        res = minimize(rosen, np.array([w*20,10]), method='L-BFGS-B',
               options={'disp': False})        
        Z0[w],L[w] = res.x
    return Z0,np.mean(L)                              # A list of 10 Z0 params and 1 L value here 


def plot_curves(param1,param2,boolean):            # plot the curves for the 2 methods
    
    colours = ['b-','g-','r-','c-','m-','y-','k-','maroon','tomato','magenta']
    
    xdata = np.linspace(0,50,51)
    
    if(boolean==0):                                    # method-1
        
        for w in range(1,11):           # Z0 and b for every w here 
            plt.plot(xdata, 100*func_20_params(xdata, param1[w], param2[w])/func_20_params(50, param1[-1], param2[-1]), colours[w-1],label=w)
            # scaled appropriately here
        plt.title('1st Method Plot',fontsize=18)
        plt.xlabel('Overs Remaining', fontsize=14)
        plt.ylabel('Resources Left', fontsize=14)
        plt.legend()
        plt.grid(b=True,linestyle='-', linewidth=2)
        plt.show()
    else:
        for w in range(1,11):           # Z0 and b for every w here 
            plt.plot(xdata, 100*func_11_params(xdata, param1[w], param2)/func_11_params(50, param1[-1], param2), colours[w-1],label=w)

        plt.title('2nd Method Plot',fontsize=18)
        plt.xlabel('Overs Remaining', fontsize=14)
        plt.ylabel('Resources Left', fontsize=14)
        plt.legend()
        plt.grid(b=True,linestyle='-', linewidth=2)
        plt.show()

# Save the plots here 

 

def compute_total_error_per_point(data,param1,param2,boolean): 
    # For each w compute the total error and return it 
    error = np.zeros(10)               # for each w we compute the total MSE error here 
    
    if(boolean==0):                       # method-1 here 
        for w in range(1,11):
            summation = 0
            data_w = data[w,:]          # contains 50 entries here 
            non_zero_indices = np.nonzero(data_w)
            data_w = data_w[non_zero_indices]               
            pred = func_20_params(non_zero_indices[0], param1[w], param2[w])
            error[w-1] = np.linalg.norm(pred-data_w)**2                      # MSE Here 
            
    else:                                  # method-2 here 
        for w in range(1,11):
            summation = 0
            data_w = data[w,:]          # contains 50 entries here 
            non_zero_indices = np.nonzero(data_w)
            data_w = data_w[non_zero_indices]               
            pred = func_11_params(non_zero_indices[0], param1[w], param2)
            error[w-1] = np.linalg.norm(pred-data_w)**2                      # MSE Here
        
    
    return error 
            
    
def compute_slopes(param1,param2,boolean):       # For each w for each of the 2 methods compute the slopes here 
    slopes = np.zeros(10)            # For the 10 weights here 
    if(boolean==0):                 # method-1 here 
        for w in range(1,11):
            slopes[w-1] = param1[w]*param2[w]         # Z*b for each w here 
        
    else:                           # method-2 here 
        for w in range(1,11):
            slopes[w-1] = param2       # L constant here 
        
    return slopes 
        
if __name__ == '__main__':        

    path = "C:\\Users\\bhu1d\\04_cricket_1999to2011.csv"       # put required path here 
    
    data = get_data(path)
    
    Z0_method1,b = DuckworthLewis20Params(data)
    
    print("\n 20 parameters 1st method")
    
    print(np.round(Z0_method1,2),np.round(b,2))
    
    Z0_method2,L = DuckworthLewis11Params(data)
    
    print("\n 11 parameters 2nd method")
    
    print(np.round(Z0_method2,2),np.round(L,2))
    
    plot_curves(Z0_method1,b,0)
    
    plot_curves(Z0_method2,L,1)

    slopes_1 = np.round(compute_slopes(Z0_method1,b,0),2)
    
    slopes_2 = np.round(compute_slopes(Z0_method2,L,1),2)
    
    print("\n Slopes in the 2 methods")
    
    print(slopes_1,slopes_2)

    error_1 = np.round(compute_total_error_per_point(data,Z0_method1,b,0),2)
    
    error_2 = np.round(compute_total_error_per_point(data,Z0_method2,L,1),2)

    # Error per point is also done here :: 
    
    print("\n Errors for the 2 methods")
    
    print(error_1,error_2)
    
