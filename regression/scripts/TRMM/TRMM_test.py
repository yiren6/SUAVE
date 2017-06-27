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
    TRM_opt.optimizer  = 'SLSQP'
    print 'Checking TRMM with no active constraints...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 0.46312425982939409 )
    assert( outputs[1][0] == 0.32827758789062461 )
    assert( outputs[1][1] == 0.10000826246176589 )  
    
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
    TRM_opt.optimizer  = 'SLSQP'
    print 'Checking TRMM with > constraint...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 2.6551249810501929 )
    assert( outputs[1][0] == -0.97387695099099969 )
    assert( outputs[1][1] == 1.0533935546874953 )      
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '<', 10., 1., Units.less],
        [ 'x1' , '=', 2., 1., Units.less],
         ]) 
    
    #outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    
    tr = Trust_Region()
    opt_prob.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 50
    TRM_opt.optimizer  = 'SLSQP'
    print 'Checking TRMM with = constraint...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 0.99999999653823579 )
    assert( outputs[1][0] == 2.0 )
    assert( outputs[1][1] == 4.0000019861033724 )        
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '<', 10., 1., Units.less],
        [ 'x1' , '<', .1, 1., Units.less],
         ]) 
    
    #outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    
    tr = Trust_Region()
    opt_prob.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.trust_region_max_iterations           = 50
    TRM_opt.optimizer  = 'SLSQP'
    print 'Checking TRMM with < constraint...'
    outputs = TRM_opt.optimize(opt_prob,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0][0] == 388.82343202898346 )
    assert( outputs[1][0] == -0.98721934861223404 )
    assert( outputs[1][1] == -0.98721953801645423 )      
 
    return

# ----------------------------------------------------------------------        
#   Call Main
# ---------------------------------------------------------------------- 

if __name__ == '__main__':
    main()
    plt.show()
