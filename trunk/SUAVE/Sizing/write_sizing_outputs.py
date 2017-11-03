#write_sizing_inputs.py
# Created: Jun 2016, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ---------------

import numpy as np


# ----------------------------------------------------------------------
#  write_sizing_outputs
# ----------------------------------------------------------------------


def write_sizing_outputs(output_filename, y_save, opt_inputs):

    file=open(output_filename, 'ab')
    file.write(str(opt_inputs))
    file.write(' ')
    file.write(str(y_save.tolist()))
    file.write('\n') 
    file.close()
                
    return