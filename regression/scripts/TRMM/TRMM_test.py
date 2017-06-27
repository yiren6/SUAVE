# TRMM_test.py
# 
# Created:  Jun 2017, T. MacDonald

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import SUAVE
import numpy as np
import matplotlib.pyplot as plt
from SUAVE.Core import Units
import problem_definition
import SUAVE.Optimization.Package_Setups.TRM_2.Trust_Region_Optimization as tro
from SUAVE.Optimization.Package_Setups.TRM_2.Trust_Region import Trust_Region
import SUAVE.Optimization.Package_Setups.pyopt_setup as pyopt_setup
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup
from contextlib import contextmanager
import sys, os

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():
    
    # ------------------------------------------------------------------
    #   Inactive constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '>', -10., 1., Units.less],
        [ 'x1' , '>', -50., 1., Units.less],
         ]) 
    
    #outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    
    tr = Trust_Region()
    opt_prob.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    TRM_opt.optimizer  = 'SNOPT'
    print 'Checking TRMM with no active constraints...'
    outputs = TRM_opt.optimize(opt_prob,print_output=True)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 0.4631250214633168 )
    assert( outputs[1][0] == 0.32827758789062511 )
    assert( outputs[1][1] == 0.10000833813898898 )  
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '>', 1., 1., Units.less],
        [ 'x1' , '>', -50., 1., Units.less],
         ]) 
    
    #outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    
    tr = Trust_Region()
    opt_prob.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    print 'Checking TRMM with one active constraint...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 3.9992876426030648 )
    assert( outputs[1][0] == -1.0011471271514893 )
    assert( outputs[1][1] == 1.0056114973472992 )      
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '<', -1., 1., Units.less],
        [ 'x1' , '=', 2., 1., Units.less],
         ]) 
    
    #outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    
    tr = Trust_Region()
    opt_prob.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 20
    print 'Checking TRMM with two active constraints...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 2501.0000000000182 )
    assert( outputs[1][0] == 2.0000000000000062 )
    assert( outputs[1][1] == -0.99999999999999356 )          
 
    return

# ----------------------------------------------------------------------        
#   Call Main
# ---------------------------------------------------------------------- 

if __name__ == '__main__':
    main()
    plt.show()
