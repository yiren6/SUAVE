import numpy as np
import copy
import SUAVE
import Trust_Region
try:
    import pyOpt
except:
    pass
from SUAVE.Core import Units, Data
from SUAVE.Optimization import helper_functions as help_fun
import os
import sys
import scipy as sp

class Trust_Region_Optimization(Data):
        
    def __defaults__(self):
        
        self.tag                                = 'TR_Opt'
        self.trust_region_max_iterations        = 30
        self.optimizer_max_iterations           = 30
        self.soft_convergence_tolerance         = 1E-6
        self.hard_convergence_tolerance         = 1E-6
        self.optimizer_convergence_tolerance    = 1E-6  #used in SNOPT
        self.optimizer_constraint_tolerance     = 1E-6  #used in SNOPT only
        self.difference_interval                = 1E-6  #used in evaluating high fidelity case
        self.optimizer_function_precision       = 1E-12 #used in SNOPT only
        self.trust_region_function_precision    = 1E-12
        self.optimizer_verify_level             = 0
        self.fidelity_levels                    = 2  
        self.evaluation_order                   = [1,2]       
        self.trust_region_history               = []
        self.objective_history                  = []
        self.constraint_history                 = []
        self.relative_difference_history        = [] # for soft convergence
        self.design_variable_history            = []
        self.optimizer                          = 'SNOPT'
        
    def optimize(self,problem,print_output=False):
        if print_output == False:
            devnull = open(os.devnull,'w')
            sys.stdout = devnull
            
        # History writing
        f_out = open('TRM_hist.txt','w')
        import datetime
        f_out.write(str(datetime.datetime.now())+'\n')
        self.x0_hist = []
        self.x1_hist = []
        self.obj_hi  = []        
        
        inp = problem.optimization_problem.inputs
        obj = problem.optimization_problem.objective
        con = problem.optimization_problem.constraints 
        tr = problem.trust_region
        
        # Set inputs
        nam = inp[:,0] # Names
        ini = inp[:,1] # Initials
        bnd = inp[:,2] # Bounds
        scl = inp[:,3] # Scale
        typ = inp[:,4] # Type
    
        (x,scaled_constraints,bnds,lbd,ubd,up_edge,low_edge,name) = self.scale_vals(inp, con, ini, bnd, scl)
        
        # ---------------------------
        # Trust region specific code
        # ---------------------------
        
        iterations = 0
        max_iterations = self.trust_region_max_iterations
        x = np.array(x,dtype='float')
        tr.center = x
        tr_size = tr.size

        trc = x # trust region center
        x_initial = x*1.
        
        fOpt_min = np.array([10000.])
        xOpt_min = x*1.        
        
        while iterations < max_iterations:
            iterations += 1
            
            # History writing
            f_out.write('Iteration ----- ' + str(iterations) + '\n')
            f_out.write('x0_center: ' + str(x[0]) + '\n')
            f_out.write('x1_center: ' + str(x[1]) + '\n')
            f_out.write('tr size  : ' + str(tr.size) + '\n')   
            self.x0_hist.append(x[0])
            self.x1_hist.append(x[1])
            
            xOpt = np.zeros(np.shape(x))
            fOpt = None
            gOpt = np.zeros(np.shape(scaled_constraints))
            f    = [None]*self.fidelity_levels
            df   = [None]*self.fidelity_levels
            g    = [None]*self.fidelity_levels
            dg   = [None]*self.fidelity_levels            
            
            for level in self.evaluation_order:
                problem.fidelity_level = level
                res = self.evaluate_model(problem,x,scaled_constraints)
                f[level-1]  = res[0]    # objective value
                df[level-1] = res[1]    # objective derivate vector
                g[level-1]  = res[2]    # constraints vector
                dg[level-1] = res[3]    # constraints jacobian
                # History writing
                f_out.write('Level    : ' + str(level) + '\n')
                f_out.write('f        : ' + str(res[0][0]) + '\n')
                f_out.write('df0      : ' + str(res[1][0]) + '\n')
                f_out.write('df1      : ' + str(res[1][1]) + '\n') 
                f_out.write('f for df0: ' + str(res[1][0]*self.difference_interval+res[0][0]) + '\n')
                f_out.write('f for df1: ' + str(res[1][1]*self.difference_interval+res[0][0]) + '\n')
                
            if iterations == 0:
                self.objective_history.append(f[0])
                self.constraint_history.append(g[0])
                
            # Calculate correction
            corrections = self.calculate_correction(f,df,g,dg,tr)
            
            # Calculate constraint violations
            g_violation_hi_center = self.calculate_constraint_violation(g[-1],low_edge,up_edge)
            
            # Subproblem
            tr_size = tr.size
            tr.lower_bound = np.max(np.vstack([lbd,x-tr_size]),axis=0)
            tr.upper_bound = np.min(np.vstack([ubd,x+tr_size]),axis=0)      
            
            problem.fidelity_level = 1
            
            if self.optimizer == 'SNOPT':
                opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_corrected_model, corrections=corrections,tr=tr)
                
                for ii in xrange(len(obj)):
                    opt_prob.addObj('f',f[-1]) 
                for ii in xrange(0,len(inp)):
                    vartype = 'c'
                    opt_prob.addVar(nam[ii],vartype,lower=tr.lower_bound[ii],upper=tr.upper_bound[ii],value=x[ii])    
                for ii in xrange(0,len(con)):
                    if con[ii][1]=='<':
                        opt_prob.addCon(name[ii], type='i', upper=up_edge[ii])  
                    elif con[ii][1]=='>':
                        opt_prob.addCon(name[ii], type='i', lower=low_edge[ii],upper=np.inf)
                    elif con[ii][1]=='=':
                        opt_prob.addCon(name[ii], type='e', equal=up_edge[ii])      
                        
                   
                opt = pyOpt.pySNOPT.SNOPT()       
                
                opt.setOption('Major iterations limit'     , self.optimizer_max_iterations)
                opt.setOption('Major optimality tolerance' , self.optimizer_convergence_tolerance)
                opt.setOption('Major feasibility tolerance', self.optimizer_constraint_tolerance)
                opt.setOption('Function precision'         , self.optimizer_function_precision)
                opt.setOption('Verify level'               , self.optimizer_verify_level)           
                
                outputs = opt(opt_prob, sens_type='FD',problem=problem,corrections=corrections,tr=tr)#, sens_step = sense_step)  
                if outputs[2]['value'][0] == 13:
                    feasible_flag = False
                    success_indicator = False
                else:
                    feasible_flag = True
                    success_indicator = True
                fOpt_lo = outputs[0][0,0]
                xOpt_lo = outputs[1]
                gOpt_lo = np.zeros([1,len(con)])[0]  
                for ii in xrange(len(con)):
                    gOpt_lo[ii] = opt_prob._solutions[0]._constraints[ii].value                
            elif self.optimizer == 'SLSQP':
                wrapper  = lambda x:self.evaluate_corrected_model(x,problem=problem,corrections=corrections,tr=tr)[0][0]
                # find bounds
                eqs       = []
                ieqs      = []
                ieqs_sign = []
                for ci in con:
                    if ci[1] == '=':
                        eqs.append(ci[2])
                    elif ci[1] == '>':
                        ieqs.append(ci[2])
                        ieqs_sign.append(1.)
                    else:
                        ieqs.append(ci[2])
                        ieqs_sign.append(-1.)
                eqs       = np.array(eqs)
                ieqs      = np.array(ieqs)
                ieqs_sign = np.array(ieqs_sign)
                eq_cons   = lambda x:problem.equality_constraint(x) - eqs
                ieq_cons  = lambda x:problem.inequality_constraint(x)*ieqs_sign - ieqs*ieqs_sign
                
                sense_step = 1.4901161193847656e-08
                slsqp_bnds = np.transpose(np.vstack([tr.lower_bound,tr.upper_bound]))
                outputs, fx, its, imode, smode = sp.optimize.fmin_slsqp(wrapper,x,f_eqcons=eq_cons,f_ieqcons=ieq_cons,bounds=slsqp_bnds,iter=200, epsilon = sense_step, acc  = sense_step**2, full_output=True)
                if (imode == 2 or imode == 4) or (imode == 9 and np.isnan(fx[0])):
                    feasible_flag = False
                    success_indicator = False
                else:
                    feasible_flag = True
                    success_indicator = True
                fOpt_lo = fx
                xOpt_lo = outputs
                gOpt_lo = problem.all_constraints(outputs)
            else:
                raise ValueError('Selected optimizer not implemented')
            success_flag = success_indicator            
        
            
            
            # Constraint minization ------------------------------------------------------------------------
            if feasible_flag == False:
                print 'Infeasible within trust region, attempting to minimize constraint'
                opt_prob = pyOpt.Optimization('SUAVE',self.evaluate_constraints, corrections=corrections,tr=tr,
                                              lb=low_edge,ub=up_edge)
                for ii in xrange(len(obj)):
                    opt_prob.addObj('constraint violation',0.) 
                for ii in xrange(0,len(inp)):
                    vartype = 'c'
                    opt_prob.addVar(nam[ii],vartype,lower=tr.lower_bound[ii],upper=tr.upper_bound[ii],value=x[ii])           
                opt = pyOpt.pySNOPT.SNOPT()            
                opt.setOption('Major iterations limit'     , self.optimizer_max_iterations)
                opt.setOption('Major optimality tolerance' , self.optimizer_convergence_tolerance)
                opt.setOption('Major feasibility tolerance', self.optimizer_constraint_tolerance)
                opt.setOption('Function precision'         , self.optimizer_function_precision)
                opt.setOption('Verify level'               , self.optimizer_verify_level)                 
                
                problem.fidelity_level = 1
               
                con_outputs = opt(opt_prob, sens_type='FD',problem=problem,corrections=corrections,tr=tr,
                                  lb=low_edge,ub=up_edge)#, sens_step = sense_step)
                xOpt_lo = con_outputs[1]
                new_outputs = self.evaluate_corrected_model(x, problem=problem,corrections=corrections,tr=tr)
                
                fOpt_lo = np.array([new_outputs[0][0,0]])
                gOpt_lo = np.zeros([1,len(con)])[0]   
                for ii in xrange(len(con)):
                    gOpt_lo[ii] = new_outputs[1][ii]
                
                # Constraint minization end ------------------------------------------------------------------------
                
       
            g_violation_opt_lo = self.calculate_constraint_violation(gOpt_lo,low_edge,up_edge)

            print 'fOpt_lo = ', fOpt_lo
            print 'xOpt_lo = ', xOpt_lo
            print 'gOpt_lo = ', gOpt_lo                 
            
            # Evaluate high-fidelity at optimum
            problem.fidelity_level = np.max(self.fidelity_levels)
            fOpt_hi, gOpt_hi = self.evaluate_model(problem,xOpt_lo,scaled_constraints,der_flag=False)
            
            self.objective_history.append(fOpt_hi)
            self.constraint_history.append(gOpt_hi)
            
            g_violation_opt_hi = self.calculate_constraint_violation(gOpt_hi,low_edge,up_edge)
            
            # Calculate ratio
            offset = 0.
            problem.fidelity_level = 2
            high_fidelity_center  = tr.evaluate_function(f[-1],g_violation_hi_center)
            high_fidelity_optimum = tr.evaluate_function(fOpt_hi,g_violation_opt_hi)
            low_fidelity_center   = tr.evaluate_function(f[-1],g_violation_hi_center)
            low_fidelity_optimum  = tr.evaluate_function(fOpt_lo,g_violation_opt_lo)
            if ( np.abs(low_fidelity_center-low_fidelity_optimum) < self.trust_region_function_precision):
                rho = 1.
            else:
                rho = (high_fidelity_center-high_fidelity_optimum)/(low_fidelity_center-low_fidelity_optimum)
            
            # Soft convergence test
            if( np.abs(fOpt_hi) <= self.trust_region_function_precision and np.abs(f[-1]) <= self.trust_region_function_precision ):
                relative_diff = 0
            elif( np.abs(fOpt_hi) <= self.trust_region_function_precision):
                relative_diff = (fOpt_hi - f[-1])/f[-1]
            else:
                relative_diff = (fOpt_hi - f[-1])/fOpt_hi
            self.relative_difference_history.append(relative_diff)
            diff_hist = self.relative_difference_history
            
            #ind1 = max(0,iterations-1-tr.soft_convergence_limit)
            #ind2 = len(diff_hist) - 1
            #converged = 1
            #while ind2 >= ind1:
                #if( np.abs(diff_hist[ind1]) > self.soft_convergence_tolerance ):
                    #converged = 0
                    #break
                #ind1 += 1
            #if( converged and len(self.relative_difference_history) >= tr.soft_convergence_limit):
                #f_out.write('soft convergence reached')
                #f_out.close()
                #all_data = np.zeros([len(self.trust_region_history),6])
                #for ii in range(len(self.trust_region_history)):
                    #all_data[ii,:] = np.array([self.trust_region_history[ii][0][0],self.trust_region_history[ii][0][1],\
                                               #self.trust_region_history[ii][1],self.design_variable_history[ii][0][0],\
                                               #self.design_variable_history[ii][0][1],self.design_variable_history[ii][1][0]])      
                #np.save('all_TRM_data.npy',all_data)                
                #np.save('TRM_cons_data.npy',np.hstack([self.constraint_history]))
                #print 'Soft convergence reached'
                #if print_output == False:
                    #sys.stdout = sys.__stdout__              
                #return outputs     
            
            # Acceptance Test
            accepted = 0
            if( fOpt_hi < f[-1] ):
                print 'Trust region update accepted since objective value is lower\n'
                accepted = 1
            elif( g_violation_opt_hi < g_violation_hi_center ):
                print 'Trust region update accepted since nonlinear constraint violation is lower\n'
                accepted = 1
            else:
                print 'Trust region update rejected (filter)\n'        
            
            # Update Trust Region Size
            print tr
            tr_size_previous = tr.size
            tr_action = 0 # 1: shrink, 2: no change, 3: expand
            if( not accepted ): # shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)        
            elif( rho < 0. ): # bad fit, shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)
            elif( rho <= tr.contract_threshold ): # okay fit, shrink trust region
                tr.size = tr.size*tr.contraction_factor
                tr_action = 1
                print 'Trust region shrunk from %f to %f\n\n' % (tr_size_previous,tr.size)
            elif( rho <= tr.expand_threshold ): # pretty good fit, retain trust region
                tr_action = 2
                print 'Trust region size remains the same at %f\n\n' % tr.size
            elif( rho <= 1.25 ): # excellent fit, expand trust region
                tr.size = tr.size*tr.expansion_factor
                tr_action = 3
                print 'Trust region expanded from %f to %f\n\n' % (tr_size_previous,tr.size)
            else: # rho > 1.25, okay-bad fit, but good for us, retain trust region
                tr_action = 2
                print 'Trust region size remains the same at %f\n\n' % tr.size  
                
            # Terminate if trust region too small
            if( tr.size < tr.minimum_size ):
                print 'Trust region too small'
                f_out.write('tr too small')
                f_out.close()
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_lo,xOpt_lo,'trust region too small')
            
            # Terminate if solution is infeasible, no change is detected, and trust region does not expand
            if( success_flag == False and tr_action < 3 and \
                np.sum(np.isclose(xOpt,x,rtol=1e-15,atol=1e-14)) == len(x) ):
                print 'Solution infeasible, no improvement can be made'
                f_out.write('Solution infeasible, no improvement can be made')
                f_out.close()
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_lo,xOpt_lo,'solution infeasible')      
            
            # History writing
            f_out.write('x0 opt   : ' + str(xOpt_lo[0]) + '\n')
            f_out.write('x1 opt   : ' + str(xOpt_lo[1]) + '\n')
            f_out.write('low obj  : ' + str(fOpt_lo) + '\n')
            f_out.write('hi  obj  : ' + str(fOpt_hi[0]) + '\n')
            self.obj_hi.append(fOpt_hi[0])
            
            self.trust_region_history.append([trc, tr_size_previous])
            self.design_variable_history.append([xOpt_lo,fOpt_hi])
            
            # hard convergence check
            if (accepted==1 and np.isclose(f[1][-1],fOpt_hi[0],rtol=self.hard_convergence_tolerance,atol=1e-12)==1):
                print 'Hard convergence reached'
                f_out.write('Hard convergence reached')
                f_out.close()
                all_data = np.zeros([len(self.trust_region_history),6])
                for ii in range(len(self.trust_region_history)):
                    all_data[ii,:] = np.array([self.trust_region_history[ii][0][0],self.trust_region_history[ii][0][1],\
                                               self.trust_region_history[ii][1],self.design_variable_history[ii][0][0],\
                                               self.design_variable_history[ii][0][1],self.design_variable_history[ii][1][0]])      
                np.save('all_TRM_data.npy',all_data)
                np.save('TRM_cons_data.npy',np.hstack([self.constraint_history]))
                if print_output == False:
                    sys.stdout = sys.__stdout__                  
                return (fOpt_lo,xOpt_lo,'convergence reached')            
            
            # Update Trust Region Center
            if accepted == 1:
                x = xOpt_lo
                tr.center = x        
            
            if fOpt_hi < fOpt_min:
                fOpt_min = fOpt_hi*1.
                xOpt_min = xOpt_lo*1.            
            
            print iterations
            print x
            print fOpt_hi
            aa = 0
        
        f_out.write('Max iteration limit reached')
        f_out.close()
        all_data = np.zeros([len(self.trust_region_history),6])
        for ii in range(len(self.trust_region_history)):
            all_data[ii,:] = np.array([self.trust_region_history[ii][0][0],self.trust_region_history[ii][0][1],\
                                       self.trust_region_history[ii][1],self.design_variable_history[ii][0][0],\
                                       self.design_variable_history[ii][0][1],self.design_variable_history[ii][1][0]])  
        np.save('all_TRM_data.npy',all_data)
        np.save('TRM_cons_data.npy',np.hstack([self.constraint_history]))
        print 'Max iteration limit reached'
        if print_output == False:
            sys.stdout = sys.__stdout__          
        return (fOpt_lo,xOpt_lo,'maximum iterations reached')
            
        
    def evaluate_model(self,problem,x,cons,der_flag=True):
        f  = np.array(0.)
        g  = np.zeros(np.shape(cons))
        df = np.zeros(np.shape(x))
        dg = np.zeros([np.size(cons),np.size(x)])
        
        
        f  = problem.objective(x)
        g  = problem.all_constraints(x)
        if der_flag == False:
            return f,g
        
        # build derivatives
        fd_step = self.difference_interval

        for ii in xrange(len(x)):
            x_fd = x*1.
            x_fd[ii] = x_fd[ii] + fd_step
            obj = problem.objective(x_fd)
            grad_cons = problem.all_constraints(x_fd)

            df[ii] = (obj - f)/fd_step

            for jj in xrange(len(cons)):
                
                dg[jj,ii] = (grad_cons[jj] - g[jj])/fd_step   
     
                         
        
        return (f,df,g,dg)


    def evaluate_corrected_model(self,x,problem=None,corrections=None,tr=None):
        #duplicate_flag, obj, gradient = self.check_for_duplicate_evals(x)
        duplicate_flag = False
        if duplicate_flag == False:
            obj   = problem.objective(x)
            const = problem.all_constraints(x).tolist()
            #const = []
            fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
            
            A, b = corrections
            x0   = tr.center
            
            obj   = obj + np.dot(A[0,:],(x-x0))+b[0]
            const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
            const = const.tolist()
        
            print 'Inputs'
            print x
            print 'Obj'
            print obj
            print 'Con'
            print const
            
        return obj,const,fail
    
    def evaluate_constraints(self,x,problem=None,corrections=None,tr=None,lb=None,ub=None):
        #duplicate_flag, obj, gradient = self.check_for_duplicate_evals(x)
        duplicate_flag = False
        if duplicate_flag == False:
            obj   = problem.objective(x)
            const = problem.all_constraints(x).tolist()
            #const = []
            fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
            
            A, b = corrections
            x0   = tr.center
            
            obj   = obj + np.dot(A[0,:],(x-x0))+b[0]
            const = const + np.matmul(A[1:,:],(x-x0))+b[1:]
            const = const.tolist()
            
            obj = self.calculate_constraint_violation(const,lb,ub)
            const = None
            
            print 'Inputs'
            print x
            print 'Obj'
            print obj
            print 'Con'
            print const            
            
        return obj,const,fail    
        
        
    def calculate_constraint_violation(self,gval,lb,ub):
        gdiff = []
  
        for i in range(len(gval)):
            if len(lb) > 0:
                if( gval[i] < lb[i] ):
                    gdiff.append(lb[i] - gval[i])
            if len(ub) > 0:    
                if( gval[i] > ub[i] ):
                    gdiff.append(gval[i] - ub[i])
    
        return np.linalg.norm(gdiff) # 2-norm of violation  
    
    def calculate_correction(self,f,df,g,dg,tr):
        nr = 1 + g[0].size
        nc = df[0].size
            
        A = np.empty((nr,nc))
        b = np.empty(nr)
            
        # objective correction
        A[0,:] = df[1] - df[0]
        b[0] = f[1] - f[0]
            
        # constraint corrections
        A[1:,:] = dg[1] - dg[0]
        b[1:] = g[1] - g[0]
            
        corr = (A,b)
        
        
        return corr        
    
    def scale_vals(self,inp,con,ini,bnd,scl):
        
        # Pull out the constraints and scale them
        bnd_constraints = help_fun.scale_const_bnds(con)
        scaled_constraints = help_fun.scale_const_values(con,bnd_constraints)

        x   = ini/scl        
        lbd  = []
        ubd  = []
        edge = []
        name = []
        up_edge  = []
        low_edge = []
        
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
        up_edge  = np.array(up_edge)         
        low_edge = np.array(low_edge)        
        bnds = np.zeros((len(inp),2))
        for ii in xrange(0,len(inp)):
            # Scaled bounds
            bnds[ii] = (bnd[ii][0]/scl[ii]),(bnd[ii][1]/scl[ii])
        
        return (x,scaled_constraints,bnds,lbd,ubd,up_edge,low_edge,name)