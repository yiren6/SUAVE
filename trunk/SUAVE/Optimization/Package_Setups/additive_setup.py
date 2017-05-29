import numpy as np
import copy
import SUAVE
import pyOpt
import sklearn
from sklearn import gaussian_process
from SUAVE.Core import Units, Data
from SUAVE.Optimization import helper_functions as help_fun
from SUAVE.Methods.Utilities.latin_hypercube_sampling import latin_hypercube_sampling
from scipy.stats import norm

def Additive_Solve(problem,num_fidelity_levels=2,num_samples=10,max_iterations=10,tolerance=1e-6,opt_type='basic',num_starts=3):
    
    if num_fidelity_levels != 2:
        raise NotImplementedError
    
    # History writing
    f_out = open('add_hist.txt','w')
    import datetime
    f_out.write(str(datetime.datetime.now())+'\n')
    
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints 

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
    # need to make this into a vector of some sort that can be added later
    lbd  = []#np.zeros(np.shape(bnd[:][1]))
    ubd  = []#np.zeros(np.shape(bnd[:][1]))
    edge = []#np.zeros(np.shape(bnd[:][1]))
    name = []#[None]*len(bnd[:][1])
    up_edge  = []
    low_edge = []
    
    
    #bnd[1000]
    for ii in xrange(0,len(inp)):
        lbd.append(bnd[ii][0]/scl[ii])
        ubd.append(bnd[ii][1]/scl[ii])

    for ii in xrange(0,len(con)):
        name.append(con[ii][0])
        edge.append(scaled_constraints[ii])
        if con[ii][1]=='<':
            up_edge.append(edge[ii])
            low_edge.append(-np.inf)
        elif con[ii][1]=='>':
            up_edge.append(np.inf)
            low_edge.append(edge[ii])
            
        elif con[ii][1]=='=':
            up_edge.append(edge[ii])
            low_edge.append(edge[ii])
        
    lbd = np.array(lbd)
    ubd = np.array(ubd)
    edge = np.array(edge)
    up_edge  = np.array(up_edge)         
    low_edge = np.array(low_edge)     
    
    x_samples = latin_hypercube_sampling(len(x),num_samples,bounds=(lbd,ubd),criterion='center')
    
    # Plot samples -----------------------------------------
    import matplotlib.pyplot as plt
    fig = plt.figure("2D Test Case",figsize=(8,6))
    axes = plt.gca()
    axes.scatter(x_samples[:,0],x_samples[:,1])
    axes.set_xticks(np.linspace(lbd[0],ubd[0],num_samples+1))
    axes.set_yticks(np.linspace(lbd[1],ubd[1],num_samples+1))
    axes.grid()    
    # ------------------------------------------------------
    
    
    f = np.zeros([num_fidelity_levels,num_samples])
    g = np.zeros([num_fidelity_levels,num_samples,len(scaled_constraints)])
    
    for level in range(1,num_fidelity_levels+1):
        problem.fidelity_level = level
        for ii,x in enumerate(x_samples):
            res = evaluate_model(problem,x,scaled_constraints)
            f[level-1,ii]  = res[0]    # objective value
            g[level-1,ii,:]  = res[1]    # constraints vector
            
    fOpt_min = 10000.
    xOpt_min = x*1.
    
    converged = False
    
    for kk in range(max_iterations):
        # Build objective surrogate
        f_diff = f[1,:] - f[0,:]
        f_additive_surrogate_base = gaussian_process.GaussianProcessRegressor()
        f_additive_surrogate = f_additive_surrogate_base.fit(x_samples, f_diff)     
        
        # Build constraint surrogate
        g_diff = g[1,:] - g[0,:]
        g_additive_surrogate_base = gaussian_process.GaussianProcessRegressor()
        g_additive_surrogate = g_additive_surrogate_base.fit(x_samples, g_diff)     
        
        ## Plot Surrogates -------------------------------------------------------
        #import matplotlib.pyplot as plt
        #x1s = np.linspace(lbd[0],ubd[0],10)
        #x2s = np.linspace(lbd[1],ubd[1],10)
        #f_test = np.zeros([len(x1s),len(x2s)])
        #g_test = np.zeros([len(x1s),len(x2s)])
        #for ii,x1 in enumerate(x1s):
            #for jj,x2 in enumerate(x2s):
                #f_test[ii,jj] = f_additive_surrogate.predict([x1,x2])
                #g_test[ii,jj] = g_additive_surrogate.predict([x1,x2])
                
        #fig = plt.figure('Objective Additive Surrogate Plot')    
        #CS = plt.contourf(x2s,x1s,f_test, linewidths=2)
        #cbar = plt.colorbar(CS)
        #cbar.ax.set_ylabel('F Surrogate')
        #plt.xlabel('Aspect Ratios')
        #plt.ylabel('Wing Areas')   
        
        ### This will only plot properly if there is only one constraint
        ##fig = plt.figure('Constraint Additive Surrogate Plot')    
        ##CS = plt.contourf(x2s,x1s,g_test, linewidths=2)
        ##cbar = plt.colorbar(CS)
        ##cbar.ax.set_ylabel('G Surrogate')
        ##plt.xlabel('Aspect Ratios')
        ##plt.ylabel('Wing Areas')       
        ## -----------------------------------------------------------------------
        
        # Optimize corrected model
        
        # Chose method ---------------
        if opt_type == 'basic':
            opt_prob = pyOpt.Optimization('SUAVE',evaluate_corrected_model, \
                                      obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)       
        
            x_eval = latin_hypercube_sampling(len(x),1,bounds=(lbd,ubd),criterion='random')[0]
            
            for ii in xrange(len(obj)):
                opt_prob.addObj('f',100) 
            for ii in xrange(0,len(inp)):
                vartype = 'c'
                opt_prob.addVar(nam[ii],vartype,lower=lbd[ii],upper=ubd[ii],value=x_eval[ii])    
            for ii in xrange(0,len(con)):
                if con[ii][1]=='<':
                    opt_prob.addCon(name[ii], type='i', upper=edge[ii])
                elif con[ii][1]=='>':
                    opt_prob.addCon(name[ii], type='i', lower=edge[ii],upper=np.inf)
                elif con[ii][1]=='=':
                    opt_prob.addCon(name[ii], type='e', equal=edge[ii])      
               
            opt = pyOpt.pySNOPT.SNOPT()      
            
            problem.fidelity_level = 1
            outputs = opt(opt_prob, sens_type='FD',problem=problem, \
                          obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)#, sens_step = sense_step)  
            fOpt = outputs[0]
            xOpt = outputs[1]
            gOpt = np.zeros([1,len(con)])[0]

        elif opt_type == 'MEI':
            fstar = np.min(f[1,:])
            opt_prob = pyOpt.Optimization('SUAVE',evaluate_expected_improvement, \
                                      obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate,fstar=fstar)     

            fOpt = np.inf
            #for mm in range(num_starts):
            
                #x_eval = latin_hypercube_sampling(len(x),1,bounds=(lbd,ubd),criterion='random')[0]
                
            for ii in xrange(len(obj)):
                opt_prob.addObj('f',100) 
            for ii in xrange(0,len(inp)):
                vartype = 'c'
                opt_prob.addVar(nam[ii],vartype,lower=lbd[ii],upper=ubd[ii])#,value=x_eval[ii])    
            for ii in xrange(0,len(con)):
                if con[ii][1]=='<':
                    opt_prob.addCon(name[ii], type='i', upper=edge[ii])
                elif con[ii][1]=='>':
                    opt_prob.addCon(name[ii], type='i', lower=edge[ii],upper=np.inf)
                elif con[ii][1]=='=':
                    opt_prob.addCon(name[ii], type='e', equal=edge[ii])      
               
            opt = pyOpt.pyALPSO.ALPSO()    
            #opt.setOption('SwarmSize', value=40)
            opt.setOption('maxOuterIter',value=10)
            opt.setOption('maxInnerIter',value=6)
            opt.setOption('seed',value=1.)
            #opt.setOption('etol',value=1.)
            
            problem.fidelity_level = 1
            #expected_improvement_carpet(lbd, ubd, problem, f_additive_surrogate, g_additive_surrogate, fstar)            
            outputs = opt(opt_prob,problem=problem, \
                          obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate,fstar=fstar)#, sens_step = sense_step)  
            if outputs[0] < fOpt:
                fOpt = outputs[0]
                xOpt = outputs[1]
                gOpt = np.zeros([1,len(con)])[0]       
                      
        
        # ---------------------------------
        
        
        f = np.hstack((f,np.zeros((num_fidelity_levels,1))))
        g = np.hstack((g,np.zeros((num_fidelity_levels,1,len(gOpt)))))
        x_samples = np.vstack((x_samples,xOpt))
        for level in range(1,num_fidelity_levels+1):
            problem.fidelity_level = level
            res = evaluate_model(problem,xOpt,scaled_constraints)
            f[level-1][-1] = res[0]
            g[level-1][-1] = res[1]
            
        # History writing
        f_out.write('Iteration: ' + str(kk+1) + '\n')
        f_out.write('x0      : ' + str(xOpt[0]) + '\n')
        f_out.write('x1      : ' + str(xOpt[1]) + '\n')
        if opt_type == 'basic':
            f_out.write('expd hi : ' + str(fOpt[0]) + '\n')
        elif opt_type == 'MEI':
            f_out.write('expd imp : ' + str(fOpt) + '\n')
        f_out.write('low obj : ' + str(f[0][-1]) + '\n')
        f_out.write('hi  obj : ' + str(f[1][-1]) + '\n') 
        if kk == (max_iterations-1):
            f_diff = f[1,:] - f[0,:]
            if opt_type == 'basic':
                fOpt = f[1][-1]
            elif opt_type == 'MEI':
                opt_prob = pyOpt.Optimization('SUAVE',evaluate_corrected_model, \
                                              obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)       
            
                min_ind = np.argmin(f[1])
                x_eval = x_samples[min_ind]
            
                for ii in xrange(len(obj)):
                    opt_prob.addObj('f',100) 
                for ii in xrange(0,len(inp)):
                    vartype = 'c'
                    opt_prob.addVar(nam[ii],vartype,lower=lbd[ii],upper=ubd[ii],value=x_eval[ii])    
                for ii in xrange(0,len(con)):
                    if con[ii][1]=='<':
                        opt_prob.addCon(name[ii], type='i', upper=edge[ii])
                    elif con[ii][1]=='>':
                        opt_prob.addCon(name[ii], type='i', lower=edge[ii],upper=np.inf)
                    elif con[ii][1]=='=':
                        opt_prob.addCon(name[ii], type='e', equal=edge[ii])      
            
                opt = pyOpt.pySNOPT.SNOPT()      
            
                problem.fidelity_level = 1
                outputs = opt(opt_prob, sens_type='FD',problem=problem, \
                              obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)#, sens_step = sense_step)  
                fOpt = outputs[0]
                xOpt = outputs[1]
                gOpt = np.zeros([1,len(con)])[0] 
                f_out.write('x0_opt  : ' + str(xOpt[0]) + '\n')
                f_out.write('x1_opt  : ' + str(xOpt[1]) + '\n')                
                f_out.write('final opt : ' + str(fOpt[0]) + '\n')
            print 'Iteration Limit Reached'
            break        
            
        fOpt = f[1][-1]
            
        #if np.sum(np.isclose(xOpt_min,xOpt,rtol=1e-4,atol=1e-12))==len(x):
            #print 'Hard convergence reached'      
            #f_out.write('Hard convergence reached')
            #converged = True
            #break
        
        if np.isclose(fOpt_min,f[1][-1],rtol=tolerance,atol=1e-12)==1:
            print 'Hard convergence reached'      
            f_out.write('Hard convergence reached')
            f_diff = f[1,:] - f[0,:]
            converged = True
            if opt_type == 'MEI':
                opt_prob = pyOpt.Optimization('SUAVE',evaluate_corrected_model, \
                                              obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)       
            
                min_ind = np.argmin(f[1])
                x_eval = x_samples[min_ind]
            
                for ii in xrange(len(obj)):
                    opt_prob.addObj('f',100) 
                for ii in xrange(0,len(inp)):
                    vartype = 'c'
                    opt_prob.addVar(nam[ii],vartype,lower=lbd[ii],upper=ubd[ii],value=x_eval[ii])    
                for ii in xrange(0,len(con)):
                    if con[ii][1]=='<':
                        opt_prob.addCon(name[ii], type='i', upper=edge[ii])
                    elif con[ii][1]=='>':
                        opt_prob.addCon(name[ii], type='i', lower=edge[ii],upper=np.inf)
                    elif con[ii][1]=='=':
                        opt_prob.addCon(name[ii], type='e', equal=edge[ii])      
            
                opt = pyOpt.pySNOPT.SNOPT()      
            
                problem.fidelity_level = 1
                outputs = opt(opt_prob, sens_type='FD',problem=problem, \
                              obj_surrogate=f_additive_surrogate,cons_surrogate=g_additive_surrogate)#, sens_step = sense_step)  
                fOpt = outputs[0]
                xOpt = outputs[1]
                gOpt = np.zeros([1,len(con)])[0] 
                f_out.write('x0_opt  : ' + str(xOpt[0]) + '\n')
                f_out.write('x1_opt  : ' + str(xOpt[1]) + '\n')                
                f_out.write('final opt : ' + str(fOpt[0]) + '\n')            
            break        
            
        if f[1][-1] < fOpt_min:
            fOpt_min = f[1][-1]*1.
            xOpt_min = xOpt*1.
       
            
        pass
    
    if converged == False:
        #print 'Iteration Limit reached'
        f_out.write('Maximum iteration limit reached')
    
    np.save('x_samples.npy',x_samples)
    np.save('all_data.npy',np.vstack([x_samples[:,0],x_samples[:,1],f_diff,f[0,:],f[1,:]]))
    f_out.close()
    print fOpt,xOpt
    return (fOpt,xOpt)
    
    
def evaluate_model(problem,x,cons,der_flag=True):
    f  = np.array(0.)
    g  = np.zeros(np.shape(cons))
    
    f  = problem.objective(x)
    g  = problem.all_constraints(x)
    
    return f,g
    
def evaluate_corrected_model(x,problem=None,obj_surrogate=None,cons_surrogate=None):
    obj   = problem.objective(x)
    const = problem.all_constraints(x).tolist()
    fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
    
    obj_addition  = obj_surrogate.predict(x)
    cons_addition = cons_surrogate.predict(x)
    
    obj   = obj + obj_addition
    const = const + cons_addition
    const = const.tolist()[0]

    print 'Inputs'
    print x
    print 'Obj'
    print obj
    print 'Con'
    print const
        
    return obj,const,fail

def evaluate_expected_improvement(x,problem=None,obj_surrogate=None,cons_surrogate=None,fstar=np.inf):
    obj   = problem.objective(x)
    const = problem.all_constraints(x).tolist()
    fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
    
    obj_addition, obj_sigma   = obj_surrogate.predict(x,return_std=True)
    cons_addition, cons_sigma = cons_surrogate.predict(x,return_std=True)
    
    fhat  = obj[0] + obj_addition
    EI    = (fstar-fhat)*norm.cdf((fstar-fhat)/obj_sigma) + obj_sigma*norm.pdf((fstar-fhat)/obj_sigma)
    const = const + cons_addition
    const = const.tolist()[0]

    print 'Inputs'
    print x
    print 'Obj'
    print -EI
    print 'Con'
    print const
        
    return -np.log(EI),const,fail

def expected_improvement_carpet(lbs,ubs,problem,obj_surrogate,cons_surrogate,fstar):

    # assumes 2d

    problem.fidelity_level = 1
    linspace_num = 20
    
    x0s = np.linspace(lbs[0],ubs[0],linspace_num)
    x1s = np.linspace(lbs[1],ubs[1],linspace_num)
    
    #for ii,sweep in enumerate(sweeps):
        #for jj,twist in enumerate(twists):

            #output = problem.objective([sweep,twist])
            #summary = problem.summary    
        
    EI = np.zeros([linspace_num,linspace_num])        
        
    for ii,x0 in enumerate(x0s):
        for jj,x1 in enumerate(x1s):
            x = [x0,x1]
            obj   = problem.objective(x)
            const = problem.all_constraints(x).tolist()    
        
            obj_addition, obj_sigma   = obj_surrogate.predict(x,return_std=True)
            cons_addition, cons_sigma = cons_surrogate.predict(x,return_std=True)
            
            fhat      = obj[0] + obj_addition
            EI[jj,ii] = (fstar-fhat)*norm.cdf((fstar-fhat)/obj_sigma) + obj_sigma*norm.pdf((fstar-fhat)/obj_sigma)
            const     = const + cons_addition
            const     = const.tolist()[0]
            
            print ii
            print jj
            print 'Expected Improvement: ' + str(EI[ii,jj])
            
    import matplotlib.pyplot as plt
            
    plt.figure(1)
    CS = plt.contourf(x0s, x1s, EI, 20, linewidths=2)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Expected Improvement')
    
    num_levels = 20
    EI = np.log(EI)
    if np.min(EI[EI!=-np.inf]) > -100:
        levals = np.linspace(np.min(EI[EI!=-np.inf]),np.max(EI),num_levels)
    else:
        levals = np.linspace(-40,np.max(EI),num_levels)    
    plt.figure(2)
    CS = plt.contourf(x0s, x1s, EI, 20, linewidths=2)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Expected Improvement')    
    
    plt.show()