import sklearn
from sklearn import gaussian_process
import numpy as np
import pylab as plt

def create_drag_surrogate(opt_file='opt_point.npy',drag_file='drag_results.npy',bounds=None,analysis_type='Euler'):
    
    opt_points   = np.load(opt_file)
    drag_results = np.load(drag_file)
    
    mach_scale = np.mean(opt_points[:,0])
    cl_scale   = np.mean(opt_points[:,1])
    tc_scale   = np.mean(opt_points[:,2])
    
    if analysis_type == 'Euler':
        opt_point_scale = np.array([mach_scale,cl_scale,tc_scale])
    elif analysis_type == 'RANS':
        re_scale   = np.mean(opt_points[:,3])
        opt_point_scale = np.array([mach_scale,cl_scale,tc_scale,re_scale])
    else:
        raise ValueError('Analysis type not supported')
    
    opt_points_scaled = opt_points/opt_point_scale
    
    drag_surrogate = gaussian_process.GaussianProcessRegressor()
    drag_surrogate = drag_surrogate.fit(opt_points_scaled, drag_results)
    
    return drag_surrogate, opt_point_scale

def plot_surrogate(x_axis,y_axis,other_vals,num_points,mask_bound,opt_file,bounds,analysis_type='Euler'):
                     
    low_bounds = bounds[0]
    up_bounds  = bounds[1]
    
    opt_points   = np.load(opt_file)                
                
    mach_scale = np.mean(opt_points[:,0])
    cl_scale   = np.mean(opt_points[:,1])
    tc_scale   = np.mean(opt_points[:,2])                 
                     
    lb = dict() # lower bounds
    ub = dict() # upper bounds
    sc = dict() # scale
    label = dict()
    pos = dict()
    rg  = dict() # range of values
    
    lb['mach'] = low_bounds[0]
    lb['cl']   = low_bounds[1]
    lb['tc']   = low_bounds[2]
    
    ub['mach'] = up_bounds[0]
    ub['cl']   = up_bounds[1]
    ub['tc']   = up_bounds[2]
    
    rg['mach'] = ub['mach']-lb['mach']
    rg['cl']   = ub['cl']-lb['cl']
    rg['tc']   = ub['tc']-lb['tc']
    
    sc['mach'] = mach_scale
    sc['cl']   = cl_scale
    sc['tc']   = tc_scale
    
    pos['mach'] = 0
    pos['cl']   = 1
    pos['tc']   = 2
    
    label['mach'] = 'Mach Number'
    label['cl']   = 'Section Lift Coefficient'
    label['tc']   = 'Thickness to Chord'
    
    if analysis_type == 'RANS':
        re_scale    = np.mean(opt_points[:,3])                            
        lb['re']    = low_bounds[3]
        ub['re']    = up_bounds[3]
        rg['re']    = ub['re']-lb['re']
        sc['re']    = re_scale
        pos['re']   = 3
        label['re'] = 'Reynolds Number'      
    
    other_vals_scaled = dict()
    for key, val in other_vals.iteritems():
        if other_vals[key] > ub[key] or other_vals[key] < lb[key]:
            print 'Warning: value is out of analysis bounds for item: ' + label[key]
        other_vals_scaled[key] = other_vals[key]/sc[key]
    
    xs = np.linspace(lb[x_axis],ub[x_axis],num_points)
    ys = np.linspace(lb[y_axis],ub[y_axis],num_points)
    
    x_opt_points = opt_points[:,pos[x_axis]]
    y_opt_points = opt_points[:,pos[y_axis]]
    
    other_pos = dict()
    masks        = dict()
    final_mask   = np.ones_like(x_opt_points)
    for key, val in pos.iteritems():
        if val != pos[x_axis] and val != pos[y_axis]:
            other_pos[key] = val
            key_points = opt_points[:,pos[key]]
            masks[key] = np.logical_and(key_points<other_vals[key]+mask_bound*rg[key],\
                                        key_points>other_vals[key]-mask_bound*rg[key])
            final_mask = np.logical_and(final_mask,masks[key])
    
    xs_scaled = xs/sc[x_axis]
    ys_scaled = ys/sc[y_axis]
    
    xs_mesh,ys_mesh = np.meshgrid(xs,ys)
    xs_mesh_scaled,ys_mesh_scaled = np.meshgrid(xs_scaled,ys_scaled)
    
    drag_carpet = np.zeros(np.shape(xs_mesh))
    
    for jj in range(len(xs)):
        for ii in range(len(ys)):
            if analysis_type == 'Euler':
                prediction_point = np.array([[1.,1.,1.]]) # this gives the average value for each since points were scaled
            elif analysis_type == 'RANS':
                prediction_point = np.array([[1.,1.,1.,1.]]) # this gives the average value for each since points were scaled
            else:
                raise ValueError('Analysis type not supported')
            prediction_point[0,pos[x_axis]] = xs_mesh_scaled[ii,jj]
            prediction_point[0,pos[y_axis]] = ys_mesh_scaled[ii,jj]
            for key in other_pos:
                prediction_point[0,pos[key]] = other_vals_scaled[key]
            drag_carpet[ii,jj] = drag_surrogate.predict(prediction_point)
            
    fig = plt.figure(label[y_axis] + ' v. ' + label[x_axis]) 
    levals = np.linspace(np.min(drag_carpet),np.max(drag_carpet),41)
    plt_handle = plt.contourf(xs_mesh,ys_mesh,drag_carpet,levels=levals)
    #plt.clabel(plt_handle, inline=1, fontsize=10)
    cbar = plt.colorbar()
    x_scatter = x_opt_points[final_mask]
    y_scatter = y_opt_points[final_mask]
    plt.scatter(x_scatter,y_scatter,s=3.,c='tomato')
    plt.xlabel(label[x_axis])
    plt.ylabel(label[y_axis])
    plt.xlim([lb[x_axis],ub[x_axis]])
    plt.ylim([lb[y_axis],ub[y_axis]])
    cbar.ax.set_ylabel('Drag')
    
    plt.show()

if __name__ == '__main__':
    analysis_type = 'RANS'
    opt_file = 'opt_point_new.npy'
    drag_file = 'drag_results.npy'
    low_bounds     = np.array([.7,.05,.03,1e7])
    up_bounds      = np.array([.9,1.,.16,3e7]) 
    bounds = (low_bounds,up_bounds)
    drag_surrogate, opt_point_scale = create_drag_surrogate(opt_file=opt_file,drag_file=drag_file,bounds=None,analysis_type=analysis_type)
    
    test_point = np.array([[8.e-01, 2.e-01, 1.e-01, 1.4e+07]])
    scaled_test_point = test_point/opt_point_scale 
    drag_est = drag_surrogate.predict(scaled_test_point)
    print drag_est    
    
    # Bounds Sweep Generalized
    # mach, re, cl, tc
    other_vals = dict()
    x_axis = 'mach'
    y_axis = 're'
    other_vals['tc'] = .1
    other_vals['cl'] = .3
    num_points = 20
    mask_bound = .08 # .1 is up to 10% away from mean, scales with bound^2 
                        # since there are two values that are checked
                        
    plot_surrogate(x_axis, y_axis, other_vals, num_points, mask_bound, opt_file, bounds,analysis_type=analysis_type)