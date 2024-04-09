from PEETMotiveList import *
from PEETModelParser import *
from PEETParticleAnalysis import pcle_dist_from_nbr
from numpy.random import random_integers
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/gpfs/cssb/user/vasishtd/nec_restart/fromjohn')


def simpleTest():
    # In[1]:
    # Test a spherical patch with curvature 0.0138 and normal along z.
    d = 12.0 / np.sqrt(2.0)
    X = np.array([[-d, -d, -1], [d, d, -1], [-d, d, -1], [d, -d, -1], [0, 0, 0]])
    iRef = 4;
    vOut = np.array([0, 0, 1])+np.random.rand(3);
    [rslt, v2] = fitTwoCurvatures(X, iRef, vOut);
    print('Found solution with RMS error ' + str(rslt.fun))
    print('  vOut = ' + str(rslt.x[2:5]))
    print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
    print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
    print()

    # In[2]:
    # Same as above but with added noise. Sometimes this converges well, sometimes not.
    X2 = X + 0.1 * np.random.randn(np.shape(X)[0], np.shape(X)[1])
    [rslt, v2] = fitTwoCurvatures(X2, iRef, vOut);
    print('Found solution with RMS error ' + str(rslt.fun))
    print('  vOut = ' + str(rslt.x[2:5]))
    print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
    print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
    print()

    # In[3]:
    # Varying positive curvature... i.e. 2 different curvatures.
    X[0, 2] = X[0, 2] + 0.5
    X[1, 2] = X[1, 2] + 0.5
    [rslt, v2] = fitTwoCurvatures(X, iRef, vOut);
    print('Found solution with RMS error ' + str(rslt.fun))
    print('  vOut = ' + str(rslt.x[2:5]))
    print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
    print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
    print()

    # In[4]:
    # Finally, try an example with negative curvature.
    X[0, 2] = X[0, 2] + 1.5
    X[1, 2] = X[1, 2] + 1.5
    [rslt, v2] = fitTwoCurvatures(X, iRef, vOut);
    print('Found solution with RMS error ' + str(rslt.fun))
    print('  vOut = ' + str(rslt.x[2:5]))
    print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
    print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
    print()

def csv_mod_test_with_vout(csvfile, modfile, maxnbrs=6, max_dist=50):
    from fit2CurvaturesWithVariableVout import fitTwoCurvatures, fittedTwoCurvatureSurface
    # Get particle positions
    a = PEETmodel(modfile)
    b = a.get_all_points()

    csv = PEETMotiveList(csvfile)
    nv = csv.angles_to_norm_vec(dummy=[0,0,-1])

    all_rslts = []
    
    # Get maxnbr nearest neighbours
    dists, nbrs = pcle_dist_from_nbr(csv, a, 1, maxnbrs)
    
    # Pick 20 random particles from model
    #for w in random_integers(0, len(nv)-1, 10):
    for w in range(len(nv)):
        print(str(w))
        these_rslts = []
        # Add original particle to neighbours
        new_nbrs = np.concatenate(([w], nbrs[w]))
        # Remove one pcle to get error change
        for r in range(1, len(new_nbrs)):
            new_nbrs_1rem = np.delete(new_nbrs, r)
            # Calculate curvatures
            rslt, v2 = fitTwoCurvatures(b[new_nbrs_1rem], 0, nv[w].to_array())
            
            new_nbrs_justcheck = np.array([w, new_nbrs[r]])
            xhat = fittedTwoCurvatureSurface(rslt.x, b[new_nbrs_justcheck], 0)
            delta = xhat - b[new_nbrs_justcheck]
            check = np.sqrt(np.trace(np.dot(delta, delta.T)))
            
##            print(str(w))
##            print(str(check))
##            print('Found solution with RMS error ' + str(rslt.fun))
##            print('  Original vOut = ' + str(nv[w].to_array()))
##            print('  vOut = ' + str(rslt.x[2:5]))
##            print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
##            print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
##            print('\n')
            these_rslts.append(np.array([w, check, rslt.fun, min(rslt.x[:2]), max(rslt.x[:2]),\
                                         rslt.x[2], rslt.x[3], rslt.x[4]]))
        all_rslts.append(np.array(these_rslts))
    return np.array(all_rslts)


def csv_mod_test(csvfile, modfile, maxnbrs=6, max_dist=50):
    from fit2Curvatures import fitTwoCurvatures, fittedTwoCurvatureSurface
    # Get particle positions
    a = PEETmodel(modfile)
    b = a.get_all_points()

    csv = PEETMotiveList(csvfile)
    nv = csv.angles_to_norm_vec()

    all_rslts = []
    
    # Pick 20 random particles from model
    #for w in random_integers(0, len(nv)-1, 5):
    for w in range(len(nv)):
        these_rslts = []
        # Get maxnbr nearest neighbours
        dists, nbrs = pcle_dist_from_nbr(csv, a, 1, maxnbrs)
        # Add original particle to neighbours
        new_nbrs = np.concatenate(([w], nbrs[w]))
        #Remove one and check result
        for r in range(1, len(new_nbrs)):
            new_nbrs_1rem = np.delete(new_nbrs, r)
            # Calculate curvatures
            rslt, v2 = fitTwoCurvatures(b[new_nbrs_1rem], 0, nv[w].to_array())
            
            new_nbrs_justcheck = np.array([w, new_nbrs[r]])
            xhat = fittedTwoCurvatureSurface(rslt.x, b[new_nbrs_justcheck], 0, nv[w].to_array())
            delta = xhat - b[new_nbrs_justcheck]
            check = np.sqrt(np.trace(np.dot(delta, delta.T)))
            
            print(str(w))
            print(str(check))
            print('Found solution with RMS error ' + str(rslt.fun))
            print('  Original vOut = ' + str(nv[w].to_array()))
            print('  vOut = ' + str(rslt.x[2:5]))
            print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
            print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
            print('\n')
            these_rslts.append(np.array([w, check, rslt.fun, min(rslt.x[:2]), max(rslt.x[:2]), \
                                         nv[w].to_array()[0], nv[w].to_array()[1], nv[w].to_array()[2]]))
        all_rslts.append(np.array(these_rslts))
    return np.array(all_rslts)



def whole_thing_test(csvfile, modfile):
    from fit2CurvaturesWithVariableVout import fitTwoCurvatures, fittedTwoCurvatureSurface
    # Get particle positions
    a = PEETmodel(modfile)
    b = a.get_all_points()

    csv = PEETMotiveList(csvfile)
    nv = csv.angles_to_norm_vec()
    all_rslts = []
    
    # Pick 20 random particles from model
    #for w in random_integers(0, len(nv)-1, 20):
    for w in range(len(nv)):
        rslt, v2 = fitTwoCurvatures(b, w, nv[w].to_array())         
        print(str(w))
        print('Found solution with RMS error ' + str(rslt.fun))
        print('  Original vOut = ' + str(nv[w].to_array()))
        print('  vOut = ' + str(rslt.x[2:5]))
        print('  k1 = ' + str(rslt.x[0]) + ' with orientation ' + str(rslt.x[5:]))
        print('  k2 = ' + str(rslt.x[1]) + ' with orientation ' + str(v2))
        print('\n')
        all_rslts.append(np.array([w, rslt.fun, min(rslt.x[:2]), max(rslt.x[:2]),\
                                         rslt.x[2], rslt.x[3], rslt.x[4]]))
    return np.array(all_rslts)

from pickle import *
from PEETPRMParser import PEETPRMFile
tube_prm  = PEETPRMFile('/gpfs/cssb/user/prazakvo/nec/tube_peet/all_combined/combined/run1/'\
            'bin1_5/run1/remdup02/combined/rembup10_0/'\
            't_fromIter2_remdup0.0_fromIter0_combined_fromIter0_remdup10.0.prm')

# Final models
for p in range(len(tube_prm.prm_dict['fnModParticle'])):
    csv = tube_prm.prm_dict['initMOTL'][p]
    mod = tube_prm.prm_dict['fnModParticle'][p]
    ks = csv_mod_test_with_vout(csv, mod, maxnbrs=30, max_dist=60)
    dump(ks, file('tube_60dist_curves_%d.pk'%(p,), 'wb'))
##
##csv = dire+'G31a_seg_centrepick_InitMOTL.csv'
##mod = dire+'G31a_seg_reduced.mod'
##ks = csv_mod_test_with_vout(csv, mod, max_dist=30)
##dump(ks, file('G31a_30dist_curves.pk', 'wb'))

