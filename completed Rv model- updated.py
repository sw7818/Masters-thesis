"""
Created on Mon Oct 25 17:59:26 2021

@author: sw7818
"""

#need to add in extra stuff

# creating an analytical model for pyroelectrics 


# importing relevent python packages
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import numpy as np 
import csv
import pandas as pd
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import math



#Defining relevent constants

pi = np.pi
perm_free = 8.85*10**-12

iter = 0

def model(Area, thickness, g_t, cprime, pprime, perm, C_A, tan, name,  H):# here we want to pass in the parameters of the sample.
    # relevent constants 
    Area = Area/(10**6)   #area in metres
    thickness = thickness/(10**6) # thickness in metres
    g_t = g_t *  (10**-5)
    H = H *  (10**-5)
    
    
    cprime = cprime * (10**6)  
    pprime = pprime /(10**6)   
    C_A = C_A * (10**-12)
    Rg = (10**11)
    # relevent equations 
    perm_0 = perm * perm_free
    volume = Area * thickness  
    x_start = -3
    x_finish = 3
    step = 0.05
    n = 1
    # Amplifier characteristics - sources of noise
    C_E = (perm_0*Area)/thickness
    C = C_E + C_A
    Tt = H/g_t
    k = 1.38*(10**-23)
    va = 1*(10**-8)
    z = 4*(10**-7)
    Ia = 2.4*(10**-16)
    
    

    
    f = np.arange(x_start,x_finish, step) # this is my logged frequency range
    frequencies = []   # empty frequencies which will be appended to 
    for t in range(len(f)):
        freq = 10**(f[t])  # unlogging the required frequencies 
        frequencies.append(freq)
        
    #noise sources and voltage response lists  
    johnson = []
    thermal = []
    Amp1 = []
    Amp2 = []
    Rvalues = []
    tot_Ns = []
    JAC = []
    JDC = []
    D_stars = []
    for x in range(len(f)):
        w = frequencies[x]*2*pi
        Rv = (Rg*n*pprime*Area*w)/((g_t*(1+(w)**2*((Tt)**2))**0.5)*((1+(w)**2*((Rg*C)**2))**0.5))
        # this is the Rv equation 
        
        Rvalues.append(np.log10(Rv))  # appending the required logged value
    
        gAC=  w*C_E*tan
        gDC=  1/Rg
        g = 1/(Rg+w*C_E*tan)
        
        Admit = math.sqrt(g**2+(w*C)**2)
        #Johnson current 
        #AC J current
        ACC = math.sqrt(4*k*300*gAC)
        VAC = ACC/Admit
        JAC.append(np.log10(VAC))
        
        #DC J current 
        DCC = math.sqrt(4*k*300*gDC)
        VDC = DCC/Admit
        JDC.append(np.log10(VDC))
        
        
        V_J =  math.sqrt((VAC**2) + (VDC**2))

         
        johnson.append(np.log10(V_J))
        # Thermal Noise 
        V_T = (Rv/n)*((4*k*(300**2)*g_t)**0.5)
        #V_T = 6.65*(10**-11)
        thermal.append(np.log10(V_T)) 
        
        # Amplifier Noise 
        # Amplifier voltage noise
        V_A = (((va)**2) + (z**2)/w)**0.5
        #V_A = 5.88*(10**-12)
        Amp1.append(np.log10(V_A)) 
        # Amplifier current noise
        V_I = (Ia*Rg)/((1+(w**2)*((Rg*C)**2))**0.5)
        #V_I = 1.46*(10**-10)
        Amp2.append(np.log10(V_I)) 
        
        
        tot_N2 = V_J**2 + V_T**2 + V_A**2 + V_I**2
        tot_N = math.sqrt(tot_N2)
        tot_Ns.append(np.log10(tot_N))
        D_star = (Rv/tot_N)*(math.sqrt(Area)*100)
        D_stars.append(np.log10(D_star)) 
        
        
    print(JDC)    
    return(f, Rvalues, D_stars, tot_Ns, JAC, JDC, thermal, Amp1, Amp2) 
    

def asign():
    global a 
    global c 
    global d 
    global e 
    global f 
    global g 
    global h
    global name
    global iter
    
    
    #writing user information to data csv
    
    
    a = entry_1.get()
    c = entry_3.get()
    d = entry_4.get()
    
    e = entry_5.get()
    f = entry_6.get()
    Area = float(e) * float(f)
    
    g = entry_7.get()
    h = entry_8.get()
    i = entry_9.get()
    j = entry_10.get()
    name = entry_name.get()
    H = entry_11.get()
    
    header = ['Name of config', 'p_prime', 'c_prime', 'G_T', 'Area', 'Thickness', 'C_A', 'Permittivity', 'loss_tan', 'H']
    
    data = [name, a, c, d, Area, i, g, h, j, H]
    if iter == 0:
        
        with open('data.csv', 'w+', encoding='UTF8') as f:
            iter =  iter + 1 
            writer = csv.writer(f)
            writer.writerow(header)
        
    
    with open('data.csv', 'a+', encoding='UTF8') as f:
        
        # here we want to check if there is an entry with that name.....
        # if there isnt we add it else no add....
        
        writer = csv.writer(f)
        writer.writerow(data)
        reader = csv.reader(f)
        
        
        

        
        
def plot():
    # Read CSV to get data
    df = pd.read_csv("data.csv")
    print('Number of entrees:  ', df.shape[0])
    names = df['Name of config']
    names = pd.Series.tolist(names)
    L =0
    for x in range(df.shape[0]):
        # Aquiring info from CSV
        name = str(df.iloc[x]['Name of config'])
        p_prime = float(df.iloc[x]['p_prime'])
        c_prime = float(df.iloc[x]['c_prime'])
        G_T = float(df.iloc[x]['G_T'])
        Area = float(df.iloc[x]['Area'])
        Thickness = float(df.iloc[x]['Thickness'])
        C_A = float(df.iloc[x]['C_A'])
        Permittivity = int((df.iloc[x]['Permittivity']))
        tan = float(df.iloc[x]['loss_tan'])
        H = float(df.iloc[x]['H'])

         # returned values (f, Rvalues, D_stars, tot_N, JAC, JDC, thermal, Amp1, Amp2)  

        xcoord = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[0]
        
        ycoord = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[1]
                
        ycoord2 = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[2]
        
        ycoordtot = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[3]
                
        ycoordJAC = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[4]
        
        ycoordJDC = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[5]
                
        ycoordT = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[6]
        
        ycoordam1 = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[7]
                
        ycoordam2 = model(Area, Thickness, G_T, c_prime, p_prime, Permittivity, C_A, tan, name, H)[8]

        # Graph Fonts 
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

        matplotlib.rc('font', **font)
        
        # Relevent figures 
        rv_fig = plt.figure(1)

        # So we're gunna want to compare with a given example
        plt.xlabel('log₁₀(frequency) (Hz)')
        plt.ylabel('log₁₀(Rv) (VW⁻¹)')
        plt.title('Responsivity Behaviour')
        plt.plot(xcoord, ycoord)
        plt.legend(names)
        
        
        
        D_fig = plt.figure(2)

        # So we're gunna want to compare with a given example
        plt.xlabel('log₁₀(frequency) (Hz)')
        plt.ylabel('log₁₀(Detectivity) (mHz¹/₂W⁻¹)')
        plt.title('Detectivity Behaviour')
        plt.plot(xcoord, ycoord2)
        plt.legend(names)
        
        Noise_fig = plt.figure(3)
        Noise_fig.set_size_inches(8, 6)


        # So we're gunna want to compare with a given example
        plt.xlabel('log₁₀(frequency) (Hz)')
        plt.ylabel('log₁₀(Noise) (VHz⁻½)')
        plt.title('Total noise distribution')
        plt.plot(xcoord, ycoordtot)
        plt.plot(xcoord, ycoordJAC)
        plt.plot(xcoord, ycoordJDC)
        plt.plot(xcoord, ycoordT)
        plt.plot(xcoord, ycoordam1)
        plt.plot(xcoord, ycoordam2)
        plt.legend(['Total noise', 'Johnson-AC','Johnson-DC','Thermal','Amp-voltage','Amp-current'])
        
        TotNoise_fig = plt.figure(4)
        plt.xlabel('log₁₀(frequency) (Hz)')
        plt.ylabel('log₁₀(Noise) (VHz⁻½)')
        plt.title('Total noise comparison')
        
        plt.plot(xcoord, ycoordtot)
        plt.legend(names)
        
    # Save figures 
    rv_fig.savefig('Rv_plot.png', bbox_inches='tight')
    
    
    D_fig.savefig('D_plot.png', bbox_inches='tight')
    
    Noise_fig.savefig('Noise_plot.png', bbox_inches='tight')
    
    TotNoise_fig.savefig('Noise_plot.png', bbox_inches='tight')
    
    
    image1 = Image.open("Rv_plot.png")
    test1 = ImageTk.PhotoImage(image1)
    
    label1 = tk.Label(image=test1)
    label1.image1 = test1
    
    image2 = Image.open("D_plot.png")
    test2 = ImageTk.PhotoImage(image2)
    
    label2 = tk.Label(image=test2)
    label2.image2 = test2
    
    image3 = Image.open("Noise_plot.png")
    test3 = ImageTk.PhotoImage(image3)
    
    label3 = tk.Label(image=test3)
    label3.image3 = test3
    
    # Position image
    label1.place(x=800, y=450)
    
    label2.place(x=800, y=50)
    
    label3.place(x=1230, y=250)
    
    
def reset():
    
    with open('data.csv', 'w+', encoding='UTF8') as f:
        global iter
        iter = 0
        writer = csv.writer(f)

# Creating front end to collect input data in the setting of a form

root = Tk()

#Providing Geometry to the form
root.geometry("1900x1000")

#Providing title to the form
root.title('Analytical Pyroelectric Model')

label_0 =Label(root,text="Analytical Pyroelectric Model", width=30,font=("bold",20))

label_0.place(x=105,y=20)

label_name =Label(root,text="Name of config", width=20,font=("bold",18))
label_name.place(x=80,y=90)

entry_name=Entry(root)
entry_name.place(x=450,y=90)

label_1 =Label(root,text="p_prime (Cm-2K-1) *(10**-6)", width=30,font=("bold",18))
label_1.place(x=80,y=130)

entry_1=Entry(root)
entry_1.place(x=450,y=130)

label_3 =Label(root,text="c_prime (Jm-3K-1) *(10**6)", width=30,font=("bold",18))
label_3.place(x=68,y=180)

entry_3=Entry(root)
entry_3.place(x=450,y=180)

label_4 =Label(root,text="G_T microWK-1 (10**-5)", width=20,font=("bold",18))
label_4.place(x=70,y=230)

entry_4=Entry(root)
entry_4.place(x=450,y=230)

label_5 =Label(root,text="Length(mm)", width=20,font=("bold",18))
label_5.place(x=80,y=280)

entry_5=Entry(root)
entry_5.place(x=450,y=280)

label_6 =Label(root,text="Width (mm)", width=20,font=("bold",18))
label_6.place(x=68,y=330)

entry_6=Entry(root)
entry_6.place(x=450,y=330)

label_7 =Label(root,text="C_A (pF)", width=20,font=("bold",18))
label_7.place(x=70,y=380)

entry_7=Entry(root)
entry_7.place(x=450,y=380)


label_8 =Label(root,text="Permittivity", width=20,font=("bold",18))
label_8.place(x=70,y=430)

entry_8=Entry(root)
entry_8.place(x=450,y=430)

label_9 =Label(root,text="Thickness (um)", width=20,font=("bold",18))
label_9.place(x=70,y=480)

entry_9=Entry(root)
entry_9.place(x=450,y=480)

label_10 =Label(root,text="loss tan", width=20,font=("bold",18))
label_10.place(x=70,y=530)

entry_10=Entry(root)
entry_10.place(x=450,y=530)

label_11 =Label(root,text="H  (10**-5)", width=20,font=("bold",18))
label_11.place(x=70,y=580)

entry_11=Entry(root)
entry_11.place(x=450,y=580)



#this creates button for submitting the info provided by the user
Button(root, text='Add' , width=20,bg="black",fg='white',command=asign).place(x=180,y=680)

Button(root, text='Plot' , width=20,bg="black",fg='white',command=plot).place(x=180,y=730)

Button(root, text='Reset' , width=20,bg="black",fg='white',command=reset).place(x=180,y=780)



root.mainloop()


