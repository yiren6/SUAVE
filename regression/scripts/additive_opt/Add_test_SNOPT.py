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
from SUAVE.Optimization.Package_Setups.additive_setup import Additive_Solve
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
    print 'Checking additive (basic) with no active constraints...'
    np.random.seed(0)
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 2.1011109357112958e-07  )
    assert( outputs[1][0] == -0.099542500786843374 )
    assert( outputs[1][1] == 2.8382451130809281e-05 )  
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '>', 1., 1., Units.less],
        [ 'x1' , '>', -50., 1., Units.less],
         ]) 
    
    print 'Checking additive (basic) with one active constraint...'
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 1.0099998952227183 )
    assert( outputs[1][0] == -5.2388778042490255e-07 )
    assert( outputs[1][1] == 1.0 )      
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '<', -1., 1., Units.less],
        [ 'x1' , '=', 2., 1., Units.less],
         ]) 
    
    print 'Checking additive (basic) with two active constraints...'
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 5.4099999282807172 )
    assert( outputs[1][0] == 2.0000000205416946 )
    assert( outputs[1][1] == -0.99999992100279655 )          
    
    
    
    
    
    
    ################# MEI ##################################################
    
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
    print 'Checking additive (MEI) with no active constraints...'
    np.random.seed(0)
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI')
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 2.1011109357112958e-07  )
    assert( outputs[1][0] == -0.099542500786843374 )
    assert( outputs[1][1] == 2.8382451130809281e-05 )  
    
    # ------------------------------------------------------------------
    #   Active constraint
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '>', 1., 1., Units.less],
        [ 'x1' , '>', -50., 1., Units.less],
         ]) 
    
    print 'Checking additive (MEI) with one active constraint...'
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI')
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 1.0099998952227183 )
    assert( outputs[1][0] == -5.2388778042490255e-07 )
    assert( outputs[1][1] == 1.0 )      
    
    # ------------------------------------------------------------------
    #   Other active constraints
    # ------------------------------------------------------------------ 
    
    opt_prob = problem_definition.problem()
    opt_prob.initialize()
    opt_prob.optimization_problem.constraints = np.array([
        [ 'x2' , '<', -1., 1., Units.less],
        [ 'x1' , '=', 2., 1., Units.less],
         ]) 
    
    print 'Checking additive (MEI) with two active constraints...'
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI')
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 5.4099999282807172 )
    assert( outputs[1][0] == 2.0000000205416946 )
    assert( outputs[1][1] == -0.99999992100279655 )              
 
    return

# ----------------------------------------------------------------------        
#   Call Main
# ---------------------------------------------------------------------- 

if __name__ == '__main__':
    main()
    plt.show()