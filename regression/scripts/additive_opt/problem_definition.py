import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
from SUAVE.Optimization import helper_functions as help_fun

class problem():
    def initialize(self):
        self.fidelity_level = 1
        self.optimization_problem = Data()
        self.optimization_problem.inputs = np.array([
            [ 'x1'                 ,  -1.  , (   -100.   ,   100.   )  ,   1.   , Units.less],
            [ 'x2'                 ,  -1.  , (   -100.   ,   100.   )  ,   1.   , Units.less],
             ])
        self.optimization_problem.objective = np.array([
            [ 'val'                ,  0.  , Units.less]
             ])
        self.optimization_problem.constraints = np.array([
            [ 'x2' , '>', -10., 1., Units.less],
            [ 'x1' , '>', -50., 1., Units.less],
             ])        
            
    def objective(self,xval):
        
        if self.fidelity_level == 1:
            obj = np.array([[xval[1]*xval[1]+xval[0]*xval[0]]])
            #obj = np.array([[90*xval[1]**2 + (0.8 - xval[0])**2]])
        elif self.fidelity_level == 2:
            obj = np.array([[xval[1]*xval[1]+(xval[0]+.1)*(xval[0]+.1)]])
            #obj = np.array([[100*(xval[1] - xval[0]**2)**2 + (1 - xval[0])**2]])
            
        return obj
    
    def all_constraints(self,xval):
        return np.array([xval[1],xval[0]])
    
    def equality_constraint(self,x = None):

        constraints = self.optimization_problem.constraints
        
        # Setup constraints  
        indices = []
        names = []
        for ii in xrange(0,len(constraints)):
            if constraints[ii][1]=='>':
                indices.append(ii)
            elif constraints[ii][1]=='<':
                indices.append(ii)
            else:
                names.append(constraints[ii][0])
        eqconstraints = np.delete(constraints,indices,axis=0)
    
        if len(eqconstraints) == 0:
            scaled_constraints = np.array([])
        else:
            constraint_values = np.ones([len(names),1])
            for ii in range(len(constraint_values)):
                if names[ii] == 'x2':
                    constraint_values[ii] = x[1]
                else:
                    constraint_values[ii] = x[0]
            scaled_constraints = help_fun.scale_const_values(eqconstraints,constraint_values)

        return scaled_constraints.flatten()
    
    def inequality_constraint(self,x = None):
        
        constraints = self.optimization_problem.constraints
        
        # Setup constraints  
        indices = []
        names = []
        for ii in xrange(0,len(constraints)):
            if constraints[ii][1]==('='):
                indices.append(ii)
            else:
                names.append(constraints[ii][0])
        iqconstraints = np.delete(constraints,indices,axis=0)
    
        if len(iqconstraints) == 0:
            scaled_constraints = np.array([])
        else:
            constraint_values = np.ones(len(names))
            for ii in range(len(constraint_values)):
                if names[ii] == 'x2':
                    constraint_values[ii] = x[1]
                else:
                    constraint_values[ii] = x[0]
            constraint_values[iqconstraints[:,1]=='<'] = -constraint_values[iqconstraints[:,1]=='<']
            bnd_constraints   = constraint_values - help_fun.scale_const_bnds(iqconstraints)
            scaled_constraints = help_fun.scale_const_values(iqconstraints,constraint_values)

        return scaled_constraints.flatten()