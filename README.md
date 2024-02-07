# SET UP
make sure you add the modules into your python path. I did it by adding a custom.pth file in .venv/lib/python3.11/site-packages with my directory, f.e.:  
/home/julian/BAJulian/magnetoelasticCoupling  

# analyze measurement:  
in measurement/postTimeGatingPlot.py uncomment in the main function (line 30 - 52) the processes you want to do:  
plot Colormap: uncomment line 42, 43 and 47  
save the resonance fields: uncomment 34   
save the values at resonance fields: uncomment 34 and 38 (select how much neigbors you want to save in line 99. set kth = 0 if you intend to fit strain   components and kth = 8 if you intend to fit alpha  

# modeling magnetoelastic Coupling:
run saw_sw_simulationYK/main.py  
be sure you comment/uncomment your desired mode parameters   

# fit magnetic parameters (you need to save the resonance fields from measurement before)
run saw_sw_simulationYK/MagParamsFit.py  
be sure you uncomment your desired mode (Rayleigh or Sezawa)  

# fit strain components
first get the values at resonance fields and set kth = 0 at line 99 (see # analyze measurement)  
run saw_sw_simulationYK/color_map_fitObject.py  
be sure you uncomment your desired mode (Rayleigh or Sezawa)  
you can norm the values, just copy and paste them into saw_sw_simulationYK/saw_simul_functions/normepsilons.py line 4-7  

# fit damping alpha
first get the values at resonance fields and set kth = 8 at line 99 (see # analyze measurement)  
uncomment line 184 of saw_sw_simulationYK/color_map_fitObject.py  
insert the fitted strain components you got in def InitialGuess() (line 40)  
run saw_sw_simulationYK/color_map_fitObject.py  
