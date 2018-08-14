## @ingroup Optimization
# carpet_plot.py
#
# Created : Feb 2016, M. Vegh 
# Modified : Feb 2017, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# -------------------------------------------
 
from SUAVE.Core import Data
import numpy as np
import matplotlib.pyplot as plt

from SUAVE.Optimization import helper_functions as help_fun

# ----------------------------------------------------------------------
#  carpet_plot
# ----------------------------------------------------------------------

## @ingroup Optimization
def line_optimization(problem, number_of_points,  plot_obj=1, plot_const=0,plot_inp=1, sweep_index=0): 
    """ Takes in an optimization problem and runs a carpet plot of the first 2 variables
        sweep_index_0, sweep_index_1 is index of variables you want to run carpet plot (i.e. sweep_index_0=0 means you want to sweep first variable, sweep_index_0 = 4 is the 5th variable)
    
        Assumptions:
        N/A
    
        Source:
        N/A
    
        Inputs:
        problem            [Nexus Class]
        number_of_points   [int]
        plot_obj           [int]
        plot_const         [int]
        sweep_index_0      [int]
        sweep_index_1      [int]
        
        Outputs:
        Beautiful Beautiful Plots!
            Outputs:
                inputs     [array]
                objective  [array]
                constraint [array]
    
        Properties Used:
        N/A
    """         

    #unpack
    idx0            = sweep_index
    opt_prob        = problem.optimization_problem
    base_inputs     = opt_prob.inputs
    names           = base_inputs[:,0] # Names
    bnd             = base_inputs[:,2] # Bounds
    scl             = base_inputs[:,3] # Scaling
    base_objective  = opt_prob.objective
    obj_name        = base_objective[0][0] #objective function name (used for scaling)
    obj_scaling     = base_objective[0][1]
    base_constraints= opt_prob.constraints
    constraint_names= base_constraints[:,0]
    constraint_scale= base_constraints[:,3]
   
    #define inputs, output, and constraints for sweep
    inputs          = np.zeros([1,number_of_points])
    obj             = np.zeros([number_of_points])
    free_inp_num    = np.shape(base_inputs)[0]
    free_inp_val    = np.zeros([free_inp_num,number_of_points])
    constraint_num  = np.shape(base_constraints)[0] # of constraints
    constraint_val  = np.zeros([constraint_num,number_of_points])
    
    
    #create inputs matrix
    inputs[0,:] = np.linspace(bnd[idx0][0], bnd[idx0][1], number_of_points)

    
    #inputs defined; now run sweep
    for i in range(0, number_of_points):
        opt_setup(problem,inputs[0,i],sweep_index=sweep_index)

        obj[i]             = problem.objective()*obj_scaling
        constraint_val[:,i]= problem.all_constraints().tolist()
        free_inp_val[:,i]  = np.array(list(problem.last_inputs[:,1])) # list strips the d-type

    if plot_obj==1:
        plt.figure(0,figsize=(4.5,3))
        plt.plot(inputs[0,:], obj, lw = 2)
        plt.xlabel(names[idx0])
        plt.ylabel(obj_name)
        
    figure_num = 0
    if plot_const==1:
        for i in range(0, constraint_num):
            plt.figure(i+1)
            plt.plot(inputs[0,:], constraint_val[i,:], lw = 2)
            plt.xlabel(names[idx0])
            plt.ylabel(constraint_names[i])
            figure_num = i+1
            
    if plot_inp==1:
        
        inp_range = range(0,free_inp_num)
        inp_range = np.delete(inp_range,sweep_index)
        
        for i in inp_range: #constraint_num):
            plt.figure(i+figure_num+1,figsize=(4,3))
            plt.plot(inputs[0,:], free_inp_val[i,:], lw = 2)
            plt.xlabel(names[idx0])
            plt.ylabel(names[i])    
            plt.minorticks_on()
            plt.axes().grid(which='both')
            plt.axes().grid(which='minor',lw=.5)
            plt.axes().grid(which='major',lw=1.5)
            
        #import plotly.graph_objs as go
        #from plotly import offline
        #trace1 = go.Carpet(
        #a = list(free_inp_val[2].flatten()),
        #b = list(free_inp_val[1].flatten()),
        #y = list(free_inp_val[0].flatten()),
        #)
        
        #data = [trace1]
        #fig = go.Figure(data = data)
        #offline.plot(fig)
            
    plt.show(block=True)      
       
        
    #pack outputs
    outputs= Data()
    outputs.inputs         = inputs
    outputs.objective      = obj
    outputs.constraint_val = constraint_val
    
    return outputs
    
def opt_setup(problem,input_0,solver='SNOPT',FD='single', sense_step=1.0E-6,  nonderivative_line_search=False,\
              sweep_index=0):
    
    inp = problem.optimization_problem.inputs
    scl = inp[:,3] # Scale  
    x_sweep_0 = input_0/scl[sweep_index]
    
    def PyOpt_Problem(problem,x):
        """ This wrapper runs the SUAVE problem and is called by the PyOpt solver.
            Prints the inputs (x) as well as the objective values and constraints.
            If any values produce NaN then a fail flag is thrown.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            problem   [nexus()]
            x         [array]
    
            Outputs:
            obj       [float]
            cons      [array]
            fail      [bool]
    
            Properties Used:
            None
        """      
       
        x_full = np.insert(x,sweep_index,x_sweep_0)           
       
        obj   = problem.objective(x_full)
        const = problem.all_constraints(x_full).tolist()
        fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
    
           
        print 'Inputs'
        print x
        print 'Obj'
        print obj
        print 'Con'
        print const
       
        return obj,const,fail    

    # Have the optimizer call the wrapper
    mywrap = lambda x:PyOpt_Problem(problem,x)
    
    inp_base = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    inp = np.delete(inp,sweep_index,axis=0)      
    
    if FD == 'parallel':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()      
    
    # Instantiate the problem and set objective
    import pyOpt
    opt_prob = pyOpt.Optimization('SUAVE',mywrap)
    for ii in xrange(len(obj)):
        opt_prob.addObj(obj[ii,0])    
       
    # Set inputs
    nam = inp[:,0] # Names
    ini = inp[:,1] # Initials
    bnd = inp[:,2] # Bounds
    scl = inp[:,3] # Scale
    typ = inp[:,4] # Type
    
    # Pull out the constraints and scale them
    bnd_constraints = help_fun.scale_const_bnds(con)
    scaled_constraints = help_fun.scale_const_values(con,bnd_constraints)
    x   = ini/scl
    
    for ii in xrange(0,len(inp)):
        lbd = (bnd[ii][0]/scl[ii])
        ubd = (bnd[ii][1]/scl[ii])
        #if typ[ii] == 'continuous':
        vartype = 'c'
        #if typ[ii] == 'integer':
            #vartype = 'i'
        opt_prob.addVar(nam[ii],vartype,lower=lbd,upper=ubd,value=x[ii])
       
    # Setup constraints  
    for ii in xrange(0,len(con)):
        name = con[ii][0]
        edge = scaled_constraints[ii]
       
        if con[ii][1]=='<':
            opt_prob.addCon(name, type='i', upper=edge)
        elif con[ii][1]=='>':
            opt_prob.addCon(name, type='i', lower=edge,upper=np.inf)
        elif con[ii][1]=='=':
            opt_prob.addCon(name, type='e', equal=edge)
    
    # Finalize problem statement and run  
    print opt_prob
    
    if solver == 'SNOPT':
        import pyOpt.pySNOPT
        opt = pyOpt.pySNOPT.SNOPT()
        CD_step = (sense_step**2.)**(1./3.)  #based on SNOPT Manual Recommendations
        opt.setOption('Function precision', sense_step**2.)
        opt.setOption('Difference interval', sense_step)
        opt.setOption('Central difference interval', CD_step)
    
    elif solver == 'COBYLA':
        import pyOpt.pyCOBYLA
        opt = pyOpt.pyCOBYLA.COBYLA() 
        
    elif solver == 'SLSQP':
        import pyOpt.pySLSQP
        opt = pyOpt.pySLSQP.SLSQP()
        opt.setOption('MAXIT', 200)
    elif solver == 'KSOPT':
        import pyOpt.pyKSOPT
        opt = pyOpt.pyKSOPT.KSOPT()
    elif solver == 'ALHSO':
        import pyOpt.pyALHSO
        opt = pyOpt.pyALHSO.ALHSO()   
    elif solver == 'FSQP':
        import pyOpt.pyFSQP
        opt = pyOpt.pyFSQP.FSQP()
    elif solver == 'PSQP':
        import pyOpt.pyPSQP
        opt = pyOpt.pyPSQP.PSQP()    
    elif solver == 'NLPQL':
        import pyOpt.pyNLPQL
        opt = pyOpt.pyNLPQL.NLPQL()    
    elif solver == 'NSGA2':
        import pyOpt.pyNSGA2
        opt = pyOpt.pyNSGA2.NSGA2(pll_type='POA') 
    elif solver == 'MIDACO':
        import pyOpt.pyMIDACO
        opt = pyOpt.pyMIDACO.MIDACO(pll_type='POA')     
    elif solver == 'ALPSO':
        import pyOpt.pyALPSO
        #opt = pyOpt.pyALPSO.ALPSO(pll_type='DPM') #this requires DPM, which is a parallel implementation
        opt = pyOpt.pyALPSO.ALPSO()
    if nonderivative_line_search==True:
        opt.setOption('Nonderivative linesearch')
    if FD == 'parallel':
        outputs = opt(opt_prob, sens_type='FD',sens_mode='pgc')
        
    elif solver == 'SNOPT' or solver == 'SLSQP':
        outputs = opt(opt_prob, sens_type='FD', sens_step = sense_step)
    
    else:
        outputs = opt(opt_prob)        
    
    return outputs