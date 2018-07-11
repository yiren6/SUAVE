## @ingroup Optimization-Package_Setups
# pyomo_setup.py
# 
# Created:  Jul 2018, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
from ipopt_setup.py import make_structure
# ----------------------------------------------------------------------
#  Ipopt_Solve
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def Pyomo_Solve(problem):
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
    
    # Pull out the basic problem
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
    
    # Scaled initials
    x0 = ini/scl
    x0 = x0.astype(float)
    
   
    # Bounds for inputs and constraints
    flbd = np.zeros_like(ini)
    fubd = np.zeros_like(ini)
    for ii in xrange(0,nvar):
        flbd[ii] = (bnd[ii][0]/scl[ii])
        fubd[ii] = (bnd[ii][1]/scl[ii])

    g_L = np.zeros_like(con)
    g_U = np.zeros_like(con)
    
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
    import pyomo   #import down here to allow SUAVE to run without the user having Ipopt
    
 
    # Create the problem
    import pyomo.environ as pyo
    model         =  Pyomo_Problem()#pyo.Concrete_Model()
    #model         = Pyomo_Problem()
    model.nvars   = nvar
    model.ncon    = ncon
    model.lower   = flbd
    model.upper   = fubd
    model.x       = pyo.Var(pyo.RangeSet(model.nvars), within pyo.Real, initialize = ini, bounds = bnd)
    model.problem = problem
   # model.A       = pyo.Set(within=Any) # define possible numbers used in math, I think
    
    def constraint_rule(model,j): #returns 0 when the constraint is violated
        #j is index for a constraint number
        problem = model.problem
        x       = model.x
        con     = problem.optimization_problem.constraints #problem constraint
        con_val = problem.all_constraints(model.x)
        edge    = con[j][2] 
        return con_val con[j][1] edge   #con[j][1] is the sign
    model.constraint = pyo.Constraint(pyo.RangeSet(model.ncons), within pyo.Real, rule = constraint_rule)
    
    #nlp = pyipopt.create(nvar, flbd, fubd, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
    
    '''
    nlp.str_option('derivative_test_print_all','yes')    
    nlp.str_option('derivative_test','first-order')


    # Solve the problem
    result = nlp.solve(x0,problem)
    nlp.close()
    '''
    
    #for j in range(len(constraints[:,0])):

    
    
    
    
    
    return result

    
try:
    from pyomo import RealOptProblem
    #instantiate a pyomo
    def class Pyomo_Problem(RealOptProblem):
        def __init__(self):
            RealOptProblem.__init__(self)
            self.grad_f   = None
            self.jac_g    = None
            self.x_grad   = [-1E7] #save value for x that gradient was computed so you don't call SUAVE twice when computing gradient and jacobian of constraints
            self.response_types = [response_enum.FunctionValue, 
                                   response_enum.Gradient,
                                   response_enum.NonlinearConstraintValues,
                                   response_enum.Jacobian]

            
            #self.lower=[0.0, -1.0, 1.0, None]
            #self.upper=[None, 0.0, 2.0, -1.0]
            #self.nvars=4
    
        def function_value(self, x):
            problem = self.problem
            out     = problem.objective(x)   
            return  out
                 
            
        def gradient(self, x):
            problem       = self.problem
            if np.isclose(x, self.x_grad):
                #already computed this point
                grad_f = self.grad_f
                jac_g  = self.jac_g
            else:
                if 'difference_interval' in problem : #check for key
                    grad_f, jac_g  = problem.finite_difference(x, problem.difference_interval)
                else:
                    grad_f, jac_g  = problem.finite_difference(x) #use default value
                self.grad_f    = grad_f
                self.jac_g     = jac_g
                self.x_grad    = x 
                
            return grad_f
            
        def jacobian(self, x):
            grad_f = self.gradient(x)
            jac_g  = self.jac_g
            return jac_g
            
        def  nonlinear_constraint_values(self,x):
            con = problem.all_constraints(x)
            return con
         
except ImportError: