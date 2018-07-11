## @ingroup Optimization-Package_Setups
# pyomo_setup.py
# 
# Created:  Jul 2018, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np

# ----------------------------------------------------------------------
#  Ipopt_Solve
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def Pyomo_Solve(problem, solver = 'baron'):
    """Solves a Nexus optimization problem using pyomo

        Assumptions:
        You can actually install pyomo on your machine
        You installed an optimizer package to go with it (such as Baron)

        Source:
        N/A

        Inputs:
        problem    [nexus()]

        Outputs:
        result     [array]

        Properties Used:
        None
    """      
    
   

    #g_L = np.zeros_like(con)
    #g_U = np.zeros_like(con)
    
    # Setup constraints
    '''
    for ii in xrange(0,len(con)):
        name = con[ii][0]
        edge = con[ii][2]
        if con[ii][1]=='<':
            g_L[ii] = -np.inf
            g_U[ii] = edge
        elif con[ii][1]=='>':
            g_L[ii] = edge
            g_U[ii] = np.inf
        elif con[ii][1]=='=':
            g_L[ii] = edge
            g_U[ii] = edge
    '''
    # Instantiate the problem and set objective

    pyomo_problem = Pyomo_Problem(problem)
    '''
    # Create the problem
    import pyomo.environ as pyo
    model         =  Pyomo_Problem(problem)#pyo.Concrete_Model()
    #model         = Pyomo_Problem()
    
    
    model.nvars   = nvar
    model.ncon    = ncon
    model.lower   = flbd
    model.upper   = fubd
   
    model.x       = pyo.Var(pyo.RangeSet(model.nvars), within=pyo.Reals, initialize = ini)
    model.problem = problem
   # model.A       = pyo.Set(within=Any) # define possible numbers used in math, I think
    
    model.constraint = pyo.Constraint(pyo.RangeSet(model.ncon), rule = constraint_rule)
    '''
    #nlp = pyipopt.create(nvar, flbd, fubd, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
    
  
    #for j in range(len(constraints[:,0])):

    opt      = pyo.SolverFactory(solver)
    results = opt.solve(pyomo_problem)
    #instance.display()
    
    return result

import pyomo
from pyomo.opt.blackbox import RealOptProblem, response_enum
import pyomo.environ as pyo
    #instantiate a pyomo problem that can be solved
    
class Pyomo_Problem(RealOptProblem):
    def __init__(self, problem):
        RealOptProblem.__init__(self)
        self.problem  = problem
        self.obj_val  = None
        self.grad_f   = None
        self.jac_g    = None
        self.x_grad   = [-1E7] #save value for x that gradient was computed so you don't call SUAVE twice when computing gradient and jacobian of constraints
        self.response_types = [response_enum.FunctionValue, 
                            response_enum.Gradient,
                            response_enum.NonlinearConstraintValues,
                            response_enum.Jacobian]
        
        
        #Pull out the basic problem
        inp = problem.optimization_problem.inputs
        obj = problem.optimization_problem.objective
        con = problem.optimization_problem.constraints
        
        # Number of input variables and constrains
        nvar = len(inp)
        ncon = len(con)
        
        # Set inputs
        ini = inp[:,1] # Initials
        bnd = inp[:,2] # Bounds
        scl = inp[:,3] # Scale
        
        #Scaled initials
        x0 = ini/scl
        x0 = x0.astype(float)
        
    
        # Bounds for inputs and constraintss
        flbd = np.zeros_like(ini)
        fubd = np.zeros_like(ini)
        
        #create pyomo model
        model         = pyo.ConcreteModel()
        model.nvars   = nvar
        model.ncon    = ncon
      
        #model.x       = pyo.Var(pyo.RangeSet(model.nvars), initialize = ini)#pyo.Var()
        model.x       = pyo.Var(pyo.RangeSet(model.nvars))#pyo.Var()
        
        for ii in xrange(0,nvar):
            flbd[ii]      = (bnd[ii][0]/scl[ii])
            fubd[ii]      = (bnd[ii][1]/scl[ii])
            model.x[ii+1] = x0[ii] #assign values
            #model.x[ii+1] = pyo.Var( bounds=(flbd[ii], fubd[ii]), initialize=ini[ii])
        print 'x here = ', model.x
        print 'dir(x) = ', dir(model.x)
        print 'x[values] = ', model.x.get_values().values()
        #model.lower   = flbd
        #model.upper   = fubd
        self.lower     = flbd
        self.higher    = fubd
        print 'self.lower = ', self.lower
        print 'self.higher = ', self.higher
        #model.x          = pyo.Var(pyo.RangeSet(model.nvars), within=pyo.Reals, initialize = ini)
        model.constraint = pyo.Constraint(pyo.RangeSet(model.ncon), rule = self.constraint_rule)
        model.problem    = problem  #assign problem to model
        self.model       = model
        
        
        
        
        #now use the SUAVE problem to create a pyomo model that can be solved
        
        #self.lower=[0.0, -1.0, 1.0, None]
        #self.upper=[None, 0.0, 2.0, -1.0]
        #self.nvars=4
    
    def function_value(self, x):
        x_here = x.get_values().values()
        problem = self.problem
        out     = problem.objective(x_here)  
        self.obj_val = out
        return  out
            
        
    def gradient(self, x):
        x_here = x.get_values().values()
        problem       = self.problem
        if np.isclose(x_here, self.x_grad):
            #already computed this point
            grad_f = self.grad_f
            jac_g  = self.jac_g
        else:
            if 'difference_interval' in problem : #check for key
                grad_f, jac_g  = problem.finite_difference(x_here, problem.difference_interval)
            else:
                grad_f, jac_g  = problem.finite_difference(x_here) #use default value
            self.grad_f    = grad_f
            self.jac_g     = jac_g
            self.x_grad    = x 
            
        return grad_f
        
    def jacobian(self, x):
        grad_f = self.gradient(x)
        jac_g  = self.jac_g
        return jac_g
        
    def  nonlinear_constraint_values(self,x):
        x_here = x.get_values().values()
        print 'x_here'
        con = problem.all_constraints(x_here)
        return con


    def constraint_rule(self, model,j): #returns 0 when the constraint is violated
        #j is index for a constraint number
        #problem = model.problem
        problem = self.problem
        x       = model.x.get_values().values()    
        con     = problem.optimization_problem.constraints #problem constraint
        con_val = problem.all_constraints(x)
  
        edge    = con[j][2] 
        symb    = con[j][1]
        if symb == '=':
            out =  con_val == edge   #con[j][1] is the sign
        elif symb == '>':
            out = con_val > edge
        elif symb == '<':
            out = con_val < edge
        return out
            
