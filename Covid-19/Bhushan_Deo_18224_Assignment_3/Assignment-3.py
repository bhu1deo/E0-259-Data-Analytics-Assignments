# Assignment-3 Covid-19 Incidence Analysis 
# Bhushan D Deo 
# M.Tech AI 18224 
# ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





def grad_descent(x,data,true_averaged_array,N):          # true averaged array is basically delta c here 

    # x = [beta,R_0,CIR_0,I_0,E_0]          # Note the order in the parameters here 

    beta,R_0,CIR_0,I_0,E_0 = x[0],x[1],x[2],x[3],x[4]

    for i in range(100):
        beta_right = beta + 0.01 
        beta_left = beta - 0.01 

        x_right = [beta_right,R_0,CIR_0,I_0,E_0]
        x_left = [beta_left,R_0,CIR_0,I_0,E_0]

        grad_beta = compute_error(x_right,data,true_averaged_array,N)[0] - compute_error(x_left,data,true_averaged_array,N)[0]         # Gradient of the loss is basically how the loss changes when we make small changes in the param

        beta = beta - (1/(100000))*grad_beta

        CIR_right = CIR_0 + 0.1
        CIR_left = CIR_0 - 0.1

        x_right = [beta,R_0,CIR_right,I_0,E_0]
        x_left = [beta,R_0,CIR_left,I_0,E_0]

        grad_CIR = compute_error(x_right,data,true_averaged_array,N)[0] - compute_error(x_left,data,true_averaged_array,N)[0]

        CIR_0 = CIR_0 - (1/(100000))*grad_CIR

        R_right = R_0 + 1 

        R_left = R_0 - 1 

        x_right = [beta,R_right,CIR_0,I_0,E_0]
        x_left = [beta,R_left,CIR_0,I_0,E_0]

        grad_R = compute_error(x_right,data,true_averaged_array,N)[0] - compute_error(x_left,data,true_averaged_array,N)[0]

        R_0 = R_0 - (1)*grad_R

        I_right = I_0 + 1 

        I_left = I_0 - 1 

        x_right = [beta,R_0,CIR_0,I_right,E_0]
        x_left = [beta,R_0,CIR_0,I_left,E_0]

        grad_I = compute_error(x_right,data,true_averaged_array,N)[0] - compute_error(x_left,data,true_averaged_array,N)[0]

        I_0 = I_0 - (1)*grad_I

        E_right = E_0 + 1 

        E_left = E_0 - 1 

        x_right = [beta,R_0,CIR_0,I_0,E_right]
        x_left = [beta,R_0,CIR_0,I_0,E_left]

        grad_E = compute_error(x_right,data,true_averaged_array,N)[0] - compute_error(x_left,data,true_averaged_array,N)[0]

        E_0 = E_0 - (1)*grad_E

    # Plot the delta prediction values here :: 
    x = [beta,R_0,CIR_0,I_0,E_0]
    print(x)
    averaged_array = compute_error(x,data,true_averaged_array,N)[1]

    plt.plot(np.linspace(0, 41, num=42),np.log(averaged_array),label="Model",linewidth=3)  
    plt.scatter(np.linspace(0, 41, num=42),np.log(true_averaged_array),c='r',label="Data")  

    plt.xlabel("Time in Days")
    plt.ylabel("Delta Infections")
    plt.legend()
    plt.title('Parameters tuned : Curve fit to Delta c')
    plt.show()


    return x            



# Error is NOT computed here 
def simulate(x,data):                              # Used to return values of S,E,I,R at the end of the simulation (fixed params)
    beta,R_0,CIR_0,I_0,E_0 = x[0],x[1],x[2],x[3],x[4]

    S_0 = N - I_0 - E_0 - R_0

    CIR_t = ((data['Tested'].values[0]/data['Tested'].values)*CIR_0)
    alpha = 1/5.8
    gamma = 1/5
    # No Vaccines for now here :: Here we simulate only for the 42 days
    S_array = np.array([S_0])
    E_array = np.array([E_0])
    I_array = np.array([I_0])
    R_array = np.array([R_0])

    for t in range(1,43):
        if(t<=29):
            delta_W = R_0/30
        else:
            delta_W = 0               # or maybe something else here 


        S_array = np.append(S_array,np.array([S_array[t-1]-beta*S_array[t-1]*I_array[t-1]/N+delta_W]))
        E_array = np.append(E_array,np.array([E_array[t-1]+beta*S_array[t-1]*I_array[t-1]/N-alpha*E_array[t-1]]))
        I_array = np.append(I_array,np.array([I_array[t-1]+alpha*E_array[t-1]-gamma*I_array[t-1]]))
        R_array = np.append(R_array,np.array([R_array[t-1]+gamma*I_array[t-1]-delta_W]))

    return S_array[-1],E_array[-1],I_array[-1],R_array,CIR_t[-1]              # All these values would be used for initializations in the prediction task 


# Error is computed here :: error is the total NORMED loss across the 42 days here 
def compute_error(x,data,true_averaged_array,N):                          # Given the true averaged array and the initial conditions, compute the model averaged array and then the error between the two here 
    beta,R_0,CIR_0,I_0,E_0 = x[0],x[1],x[2],x[3],x[4]
    # 6th one is automatically inferred here 
    S_0 = N - I_0 - E_0 - R_0
    CIR_t = ((data['Tested'].values[0]/data['Tested'].values)*CIR_0)       # Only CIR_0 is taken as initial argument here 

    alpha = 1/5.8
    gamma = 1/5
    # No Vaccines for now here :: Here we simulate only for the 42 days
    S_array = np.array([S_0])
    E_array = np.array([E_0])
    I_array = np.array([I_0])
    R_array = np.array([R_0])

    for t in range(1,43): 
        if(t<=29):
            delta_W = R_0/30             # delta_W_0 : one can also try to optimize delta_W_0 here 
        else:
            delta_W = 0               # as per the suggested logic here 

        S_array = np.append(S_array,np.array([S_array[t-1]-beta*S_array[t-1]*I_array[t-1]/N+delta_W]))
        E_array = np.append(E_array,np.array([E_array[t-1]+beta*S_array[t-1]*I_array[t-1]/N-alpha*E_array[t-1]]))
        I_array = np.append(I_array,np.array([I_array[t-1]+alpha*E_array[t-1]-gamma*I_array[t-1]]))
        R_array = np.append(R_array,np.array([R_array[t-1]+gamma*I_array[t-1]-delta_W]))


    cases = I_array/CIR_t            # To bring them back to the same scale here 
#     print(I_array)

    df = pd.DataFrame(cases, columns = ['Cases'])
    df = df.diff()
    # print(df)

    averaged_array = []

    delta_cases = df['Cases'].values[1:]
    # print(delta_cases.shape)

    for i in range(delta_cases.shape[0]):
        if(i<=6):
            averaged_array.append(delta_cases[0:i+1].mean())
        else:
            averaged_array.append(delta_cases[i-6:i+1].mean())
    averaged_array = np.array(averaged_array)

    error = 0 
    for i in range(true_averaged_array.shape[0]):
        error+=((true_averaged_array[i]-averaged_array[i])**2)

    error = error**0.5
#     error = np.linalg.norm(true_averaged_array-averaged_array)

    return error,averaged_array





def data_preprocess_train(file_path):
    """
        Training the parameters extract 16th March to 26th April of the Data here 
    """

    data_frame = pd.read_csv(file_path)


    start_row = 560-188-1
    end_row = 560-188+41                      # date slicing here 

    data = data_frame.iloc[start_row:end_row+1,:]

    data = data[['Confirmed','Tested','First Dose Administered','Recovered','Deceased']]

    data['Infections_reported'] = data['Confirmed'] - data['Recovered'] - data['Deceased']
    data = data.reset_index(drop=True)

    df = pd.DataFrame(data, columns=['Infections_reported'])    # This*CIR would be our actual infections here 
    df = df.diff()            # delta c here 


    delta_cases = df['Infections_reported'].values[1:]    # This was difference of the cases here 

    true_averaged_array = []

    for i in range(delta_cases.shape[0]):
        if(i<=6):
            true_averaged_array.append(delta_cases[0:i+1].mean())
        else:
            true_averaged_array.append(delta_cases[i-6:i+1].mean())

    true_averaged_array = np.array(true_averaged_array)                  # Average of delta c here 


    return data,true_averaged_array                       # Prescribed range required data frame returned here 


def data_preprocess_predict(file_path):
    """
        Future timestep predictions here  extract 27th April to 19th September of the data here 
    """ 
    data_frame = pd.read_csv(file_path)

    end_row = 560-188+41
    data_tot = data_frame.iloc[end_row+1:,:]

    data_tot = data_tot[['Confirmed','Tested','First Dose Administered','Recovered','Deceased']]

    data_tot['Infections_reported'] = data_tot['Confirmed'] - data_tot['Recovered'] - data_tot['Deceased']      # Infections at time t CIR needs to be taken into account here 
    data_tot = data_tot.reset_index(drop=True)

    return data_tot                     # We do not do 7 day averaging during predictions here 


def open_loop_control(S,E,I,R,CIR_0,beta,epsilon,R_0_prev_time,data_tot):          # Note that data_frame is the original data frame here 
    S_0,E_0,I_0,R_0 = S,E,I,R
    CIR_t = ((data_tot['Tested'].values[0]/data_tot['Tested'].values)*CIR_0)      # Data_tot is from 27th April here 

    alpha = 1/5.8
    gamma = 1/5
    S_array = np.array([S_0])
    E_array = np.array([E_0])
    I_array = np.array([I_0])
    R_array = np.array([R_0])

    df = pd.DataFrame(data_tot, columns=['First Dose Administered'])    # This*CIR would be our actual infections here 
    df = df.diff()

    df = df.fillna(200000)              # No NA values here 

    vaccine = np.squeeze(df.values)

    # 27th April to 20th September is close to 146 days 

    for t in range(1,146):
        if(t<=135):                          # Till 11th September no waning model here 
            delta_W = 0
        else:
            delta_W = R_0_prev_time[t-136] + epsilon*vaccine[t-136]              # or maybe something else here 


        if(S_array[t-1]-beta*S_array[t-1]*I_array[t-1]/N+delta_W-epsilon*vaccine[t-1]<0):
            S_array = np.append(S_array,np.array([0]))
        else:
            S_array = np.append(S_array,np.array([S_array[t-1]-beta*S_array[t-1]*I_array[t-1]/N+delta_W-epsilon*vaccine[t-1]]))
        E_array = np.append(E_array,np.array([E_array[t-1]+beta*S_array[t-1]*I_array[t-1]/N-alpha*E_array[t-1]]))
        I_array = np.append(I_array,np.array([I_array[t-1]+alpha*E_array[t-1]-gamma*I_array[t-1]]))
        R_array = np.append(R_array,np.array([R_array[t-1]+gamma*I_array[t-1]-delta_W+epsilon*vaccine[t-1]]))


    cases = I_array/CIR_t

    # Plot it 

    num_pts = data_tot['Infections_reported'].values.shape[0]

    plt.plot(np.linspace(0, num_pts-1, num=num_pts),data_tot['Infections_reported'].values,label='true')
    plt.plot(np.linspace(0, num_pts-1, num=num_pts),cases,label='predictions')

    plt.xlabel("Time in Days")
    plt.ylabel("Infections")


    plt.legend()
    plt.title('Open Loop Control Predictions')
    plt.show()




    return cases                     # For prediction only the cases would be compared here 


def closed_loop_control(S,E,I,R,CIR_0,beta_init,epsilon,R_0_prev_time,data_tot):          # Note that data_frame is the original data frame here 
    S_0,E_0,I_0,R_0 = S,E,I,R
    CIR_t = ((data_tot['Tested'].values[0]/data_tot['Tested'].values)*CIR_0)      # Data_tot is from 27th April here 

    alpha = 1/5.8
    gamma = 1/5
    S_array = np.array([S_0])
    E_array = np.array([E_0])
    I_array = np.array([I_0])
    R_array = np.array([R_0])

    df = pd.DataFrame(data_tot, columns=['First Dose Administered'])    # This*CIR would be our actual infections here 
    df = df.diff()

    df = df.fillna(200000)

    vaccine = np.squeeze(df.values)

    beta = beta_init
    summa = 0
    for t in range(1,146):
        if(t%7==1):
            if(t!=1):
                if(summa>100001):
                    beta = beta_init/3
                elif(summa>25000 and summa<100000):
                    beta = beta_init/2
                elif(summa<25000 and summa>10001):
                    beta = beta_init*0.66
                else:
                    beta = beta_init    
            summa = 0                         # make it 0 once the counter completes a cycle of 7 here 


        if(t<=135):
            delta_W = 0
        else:
            delta_W = R_0_prev_time[t-136] + epsilon*vaccine[t-136]               # or maybe something else here 



        S_array = np.append(S_array,np.array([S_array[t-1]-beta*S_array[t-1]*I_array[t-1]/N+delta_W-epsilon*vaccine[t-1]]))
        E_array = np.append(E_array,np.array([E_array[t-1]+beta*S_array[t-1]*I_array[t-1]/N-alpha*E_array[t-1]]))
        I_array = np.append(I_array,np.array([I_array[t-1]+alpha*E_array[t-1]-gamma*I_array[t-1]]))
        R_array = np.append(R_array,np.array([R_array[t-1]+gamma*I_array[t-1]-delta_W+epsilon*vaccine[t-1]]))

        summa+=(alpha*E_array[t-1]-gamma*I_array[t-1])/CIR_t[t-1]

        

    cases = I_array/CIR_t

    # Plot here :: 

    num_pts = data_tot['Infections_reported'].values.shape[0]

    plt.plot(np.linspace(0, num_pts-1, num=num_pts),data_tot['Infections_reported'].values,label='true')
    plt.plot(np.linspace(0, num_pts-1, num=num_pts),cases,label='predictions')

    plt.xlabel("Time in Days")
    plt.ylabel("Infections")


    plt.legend()
    plt.title('Closed Loop Control Predictions')
    plt.show()


    return cases







if __name__ == '__main__':
    # Part-1 : Tuning the parameters 

    file_path = "../COVID19_data.csv"

    # training data here :: 
    print("\nLoading data...")
    data_train,true_averaged_array = data_preprocess_train(file_path)
    # parameters obtained from the training data 
    N = 7*(10**7)           # 70 Mn here 
#     beta,R_0,CIR_0,I_0,E_0
    param_init = [1,35*N/100,12,0.38*N/100,0.5*N/100]                                  # beta,R_0,CIR_0,I_0,E_0 = x[0],x[1],x[2],x[3],x[4]
    print("\nTraining parameters...")
    parameters = grad_descent(param_init,data_train,true_averaged_array,N)            # Now we have the tuned parameters at our disposal here also plotting is done inside this function after tuning 


    # Part-2 : Future Time step prediction 
    # prediction data here ::

    data_predict = data_preprocess_predict(file_path)
    # open and closed loop control here 

    # first get S,E,I,R at the end of 42 days here 

    S,E,I,R_0_prev_time,CIR_0 = simulate(parameters,data_train)
    beta = parameters[0]

    R = R_0_prev_time[-1]       # The whole array is required to take into account the waning and the vaccination here 
    I = data_predict['Infections_reported'].values[0]*CIR_0      # Actual here 

    epsilon = 0.66
    print("\nOpen Loop Prediction...")

    cases_open_loop = open_loop_control(S,E,I,R,CIR_0,beta,epsilon,R_0_prev_time,data_predict)        # Plot also implicit here 

    print("\nClosed Loop Prediction...")
    cases_closed_loop = closed_loop_control(S,E,I,R,CIR_0,beta,epsilon,R_0_prev_time,data_predict)                 # Plotting is done here 

