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
    
    tol = 1e-8
    
    ## ------------------------------------------------------------------
    ##   Inactive constraints
    ## ------------------------------------------------------------------ 
    
    #opt_prob = problem_definition.problem()
    #opt_prob.initialize()
    #opt_prob.optimization_problem.constraints = np.array([
        #[ 'x2' , '>', -10., 1., Units.less],
        #[ 'x1' , '>', -50., 1., Units.less],
         #]) 
    
    ##outputs = pyopt_setup.Pyopt_Solve(opt_prob,solver='SNOPT')
    #print 'Checking additive (basic) with no active constraints...'
    #np.random.seed(0)
    #outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,tolerance=tol)
    
    ## ------------------------------------------------------------------
    ##   Check Results
    ## ------------------------------------------------------------------    

    #assert( outputs[0]    == 1.2165548257707792e-17  )
    #assert( outputs[1][0] == -0.10000000347984909 )
    #assert( outputs[1][1] == 2.370623809107071e-10 )  
    
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
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,tolerance=tol)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 1.0000000073474917 )
    assert( outputs[1][0] == -0.099980431783929985 )
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
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,tolerance=tol)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 5.4100013422213946 )
    assert( outputs[1][0] == 2.0000002056990813 )
    assert( outputs[1][1] == -1.0000002392737037 )          
    
    
    
    
    
    
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
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI',tolerance=tol)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == -0.00039232461266092492 )
    assert( outputs[1][0] == -0.1017825988654073 )
    assert( outputs[1][1] == 0.0021533658913842624 )  
    
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
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI',tolerance=tol)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 1.0085529184107909 )
    assert( outputs[1][0] == -0.015191935547066961 )
    assert( outputs[1][1] == 0.99999999999958955 )      
    
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
    outputs = Additive_Solve(opt_prob,max_iterations=10,print_output=False,opt_type='MEI',tolerance=tol)
    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( outputs[0]    == 5.0014822941343651 )
    assert( outputs[1][0] == 2.0000000000374905 )
    assert( outputs[1][1] == -0.99999999999996181 )              
 
    return

# ----------------------------------------------------------------------        
#   Call Main
# ---------------------------------------------------------------------- 

if __name__ == '__main__':
    main()
    plt.show()