# -*- coding: utf-8 -*-

from cPickle import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import glob
from matplotlib import ticker
import pandas as pd




def make_curve_plot(pkfile, c=0.02, outfile=None, key=False):
    data = load(file(pkfile, 'rb'))
    new_data = []
    k1 = []
    k2 = []
    for datum in data:
        clean_data = []
        for x in datum:
            if x.sum() != 0:
                clean_data.append(x)
        clean_data = np.array(clean_data)
        if len(clean_data != 0):
            k1.append(np.mean(clean_data[:,3]))
            k2.append(np.mean(clean_data[:,4]))
            
    heatmap, xedges, yedges = np.histogram2d(k1, k2, bins=50, range=[[-0.005, c], [-0.005, c]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    im = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='bone_r')

    if key:
        fig, ax = plt.subplots(figsize=(1, 6))
        fig.subplots_adjust(right=0.5)

        cmap = mpl.cm.bone_r #im.cmap
        values = np.unique(heatmap.ravel())
        norms = mpl.colors.Normalize(vmin=int(min(values)), vmax=int(max(values)))
        mappable = mpl.cm.ScalarMappable(norm=norms, cmap=cmap)
        mappable.set_array(heatmap)
        fig.colorbar(mappable,
                 cax=ax, orientation='vertical')
    
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()


def make_big_curve_plot(dire, c=0.02, outfile=None, template='*pk', key=False, apix = 14.2):
    new_data = []
    k1 = []
    k2 = []
    for x in glob.glob(dire+template):
        print(x)
        data = load(file(x, 'rb'))
        for datum in data:
            clean_data = []
            for x in datum:
                if x.sum() != 0:
                    clean_data.append(x)
            clean_data = np.array(clean_data)
            if len(clean_data != 0):
                k1.append(np.mean(clean_data[:,3]))
                k2.append(np.mean(clean_data[:,4]))
    heatmap, xedges, yedges = np.histogram2d(np.array(k1)/apix, np.array(k2)/apix, bins=50, range=[[-c, c], [-c, c]])
    
    df = pd.DataFrame(np.array(k1)/apix)
    df.to_excel('k1_' + outfile[:-4]+'.xls', index = True)
    df = pd.DataFrame(np.array(k2)/apix)
    df.to_excel('k2_' + outfile[:-4]+'.xls', index = True)
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #plt.clf()
    ff, ax = plt.subplots(figsize = (5,4),dpi = 300)
    ax.plot([-c, c],[-c, c], linestyle = 'dashed', c = 'k', alpha = 0.5)
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='bone_r')

    nticks = 11
    
    labels = ax.get_xticklabels()
    nl = np.linspace(-c, c, nticks)
    inl = np.array(np.array(np.round(1/nl, decimals = 0), dtype = int), dtype = str)
    inl[len(inl)//2] = 'inf'
    ax.set_xticklabels(inl, rotation = 45, fontsize = 11)

    ax.set_ylim(-c/10,c)
    print(inl)
    ax.set_yticklabels(inl[(nticks//2)-1:], rotation = 45, fontsize = 11)#fml




    ax.yaxis.set_major_locator(ticker.MaxNLocator(nticks//2 + 1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nticks - 1))

    
    ax.vlines(0, 0, c, linestyle = 'dotted', alpha = 0.5, color = 'k')
    ax.hlines(0, -c, 0, linestyle = 'dotted', alpha = 0.5, color = 'k')
    #ax.set_xlabel('radius of curvature (1/$\\kappa$/$\AA$)',fontsize = 12)
    ax.set_xlabel('radius of curvature ($\AA$)', fontsize = 12)
    ax.set_ylabel('radius of curvature ($\AA$)', fontsize = 12)
    
    ff.subplots_adjust(bottom = 0.2, left = 0.2)


    if key:
        fig, ax = plt.subplots(figsize=(1, 3))
        fig.subplots_adjust(right=0.18)

        cmap = mpl.cm.bone_r #im.cmap
        values = np.unique(heatmap.ravel())
        norms = mpl.colors.Normalize(vmin=int(min(values)), vmax=int(max(values)))
        mappable = mpl.cm.ScalarMappable(norm=norms, cmap=cmap)
        mappable.set_array(heatmap)
        fig.colorbar(mappable,
                 cax=ax, orientation='vertical')
    
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()
    return k1,k2
    

dire = '/gpfs/cssb/user/vasishtd/nec_restart/fromjohn/'
template = 'cyl_test2_maxdist60.pk'#'*maxdist60pix_test.pk'
outfile = 'maxdist60pix_new_test_key.png'

#make_big_curve_plot(dire, 0.02, outfile, template, key=True)

#make_curve_plot('cyl_test2_maxdist60.pk', outfile='cyltest2_justpos_key.png', key=True)
#make_curve_plot('newcyltest_60dist_curves.pk', outfile='newcyltest_justpos.png')#, key=True)
if 1:
##
    template = 'tom*maxdist60pix_test.pk' #bin 2?
    s1, s2 = make_big_curve_plot(dire, 0.025/7.1, 'sphnec_60pix_curves_b.png', template, key=False, apix = 7.1)

    #1
    dire = '/gpfs/cssb/user/prazakvo/nec/curvature/'
    template = 'tube_60dist_curves_0.pk' #remdup, 30
    k1,k2 = make_big_curve_plot(dire, 0.025/7.1, 'all_tubes.png', template, key=False, apix = 3.55)
    plt.show()
##
##    template = 'g35a_60dist_curves_*b.pk' #remdup 20
##    #tube_prm  = PEETPRMFile('/gpfs/cssb/user/prazakvo/nec/tube_peet/all_combined/combined/run1/bin1_5/run1/remdup02/combined/rembup10_0/t_fromIter2_remdup0.0_fromIter0_combined_fromIter0_remdup10.0.prm'
##
##    l1,l2 = make_big_curve_plot(dire, 0.04, 'tubenec_60pix_curves_b.png', template, key=False, apix = 7.1)
##
##    #3
##    template = 'g35a_60dist_curves_*d.pk' #20 px, shifted correctly
##    m1,m2 = make_big_curve_plot(dire, 0.04, 'tubenec_60pix_curves_b.png', template, key=False, apix = 7.1)
##    #4
##    template = 'g35a_60dist_curves_*e.pk' #30 px, shifted correctly
##    n1,n2 = make_big_curve_plot(dire, 0.04, 'tubenec_60pix_curves_b.png', template, key=False, apix = 7.1)
    #5
##    template = 'g35a_60dist_curves_*f.pk' #30 px, shifted in z only (not rotated to match sph)
##    o1,o2 = make_big_curve_plot(dire, 0.04, 'tubenec_60pix_curves_b.png', template, key=False, apix = 7.1)
##  

##
##    alpha = 0.5
##    alpha2 = 0.7
##    ff,ax = plt.subplots(1,2)
##    ax[0].hist(np.array(k2)/7.1, bins = 20, density = True, alpha = alpha,  label = '1')
##    ax[0].hist(np.array(l2)/7.1, bins = 20, density = True, alpha = alpha,  label = '2')
##    ax[0].hist(np.array(m2)/7.1, bins = 20, density = True, alpha = alpha,  label = '3')
##    ax[0].hist(np.array(n2)/7.1, bins = 20, density = True, alpha = alpha,  label = '4')
##    ax[0].hist(np.array(o2)/7.1, bins = 20, density = True, alpha = alpha,  label = '5')
##    ax[0].hist(np.array(s2)/7.1, bins = 100, density = True, alpha = alpha, label = 'sph')
##
##    ax[1].hist(np.array(k1)/7.1, bins = 20, density = True, alpha = alpha,  label = '1')
##    ax[1].hist(np.array(l1)/7.1, bins = 20, density = True, alpha = alpha,  label = '2')
##    ax[1].hist(np.array(m1)/7.1, bins = 20, density = True, alpha = alpha,  label = '3')
##    ax[1].hist(np.array(n1)/7.1, bins = 20, density = True, alpha = alpha,  label = '4')
##    ax[1].hist(np.array(o1)/7.1, bins = 20, density = True, alpha = alpha,  label = '5')
##    ax[1].hist(np.array(s1)/7.1, bins = 100, density = True, alpha = alpha,  label = 'sph')
##    plt.legend()
##    plt.show()
##
##    ff,ax = plt.subplots(1,2)
##    ax[0].hist(np.array(k2), bins = 20, density = True, alpha = alpha, label = '1')
##    ax[0].hist(np.array(l2), bins = 20, density = True, alpha = alpha, label = '2')
##    ax[0].hist(np.array(m2), bins = 20, density = True, alpha = alpha, label = '3')
##    ax[0].hist(np.array(n2), bins = 20, density = True, alpha = alpha,  label = '4')
##    ax[0].hist(np.array(o2), bins = 20, density = True, alpha = alpha,  label = '5')
##    ax[0].hist(np.array(s2), bins = 100, density = True, alpha = alpha, label = 'sph')
##    
##    ax[1].hist(np.array(k1), bins = 20, density = True, alpha = alpha, label = '1')
##    ax[1].hist(np.array(l1), bins = 20, density = True, alpha = alpha, label = '2')
##    ax[1].hist(np.array(m1), bins = 20, density = True, alpha = alpha, label = '3')
##    ax[1].hist(np.array(n1), bins = 20, density = True, alpha = alpha,  label = '4')
##    ax[1].hist(np.array(o1), bins = 20, density = True, alpha = alpha,  label = '5')
##    ax[1].hist(np.array(s1), bins = 100, density = True, alpha = alpha, label = 'sph')
##    plt.legend()
##    plt.show()
    #template = 'nascentnec_60dist_curves*.pk'
    #make_big_curve_plot(dire, 0.04, 'nascentnec_60pix_curves_b.png', template, key=False) 
if 0:
    make_big_curve_plot(dire, 0.04, 'tubenec_60pix_curves_key.png', template, key=True)
    template = 'tom*maxdist60pix_test.pk'
    make_big_curve_plot(dire, 0.04, 'sphnec_60pix_curves_key.png', template, key=True) 
    template = 'nascentnec_60dist_curves*.pk'
    make_big_curve_plot(dire, 0.04, 'nascentnec_60pix_curves_key.png', template, key=True) 

"""
dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist20novout/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist20_novout_all.png')

dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist20/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist20_vout_all.png')

dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist20_avenv/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist20_avenv_all.png')


dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist40novout/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist40_novout_all.png')

dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist40/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist40_vout_all.png')

dire = '/raid/fsj/grunewald/daven/software/fromjohn/maxdist40_avenv/'
make_big_curve_plot(dire, c=0.02, outfile='maxdist40_avenv_all.png')
"""
#for x in glob.glob(dire+'*pk'):
#    make_curve_plot(x, outfile=x[:-3]+'_plot.png')
   
