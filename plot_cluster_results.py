import pickle
import numpy as np
from compare_heuristic_script import generate_heuristic_plot
from compare_defender_utilities_script import generate_defender_utilities_plot
from compare_defender_utilities_script import generate_conditional_prob_plot
from compare_defender_utilities_script import generate_conditional_prob_plot
from number_attackers_uavs_over_gamma_script import generate_ratio_plots
import sys
import copy
import argparse


def plot_cluster_results(folders, 
                         timing=False, 
                         utilities=False, 
                         ratio=False,
                         mode='3a',
                         save=False, 
                         show=True):
    """
    title::
        plot_cluster_results

    description::
        If the methods are run on clusters to take advantage of parallel runs,
        for example to get 20 graphs, this will combine the results and then 
        call the plotting functions. Assuming 1 graph called per run. Also 
        assuming the graphs are all different (may check this with md5sum for 
        example).
    
    attributes::
        folders
            List of folders containing final pickles from the various runs, 
            i.e., each folder should have all resulting pickles from that run
        timing
            Whether folders provided are for timing experiments (3a, 3b). Note
            that in these experiments, could parallelize by num graphs AND N,
            but in this code, we only accept parallelizing by num graphs. Could
            be addressed by using np.hstack instead of np.vstack
            [default is False]
        utilities
            Whether folders provided are for utility experiments (3c, 3d, 3e). 
            Note that in these experiments, can parallelize by num graphs AND
            gamma, but in this code, we only accept parallelizing by num
            graphs. Could be addressed by using np.hstack instead of np.vstack
            [default is False]
        ratio
            Whether folders provided are for ratio experiments (3f, 3g), can 
            parallelize by num graphs amongst other loops, but in this code, we
            only accept parallelizing by num graphs [default is False]
        mode
            '3a', '3b', or 'both' for which plot to create based on the current
            resultsTiming (for ease since they cover different ranges in the 
            paper)
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is True] NOTE: do not set both save and show to True, one may be 
            blank

    author::
        Elizabeth Bondi (ebondi@g.harvard.edu)
        Hoon Oh, Haifeng Xu, Kai Wang

    disclaimer::
        This source code is provided "as is" and without warranties as to 
        performance or merchantability. The author and/or distributors of 
        this source code may have made statements about this source code. 
        Any such statements do not constitute warranties and shall not be 
        relied on by the user in deciding whether to use this source code.
      
        This source code is provided without any express or implied warranties 
        whatsoever. Because of the diversity of conditions and hardware under 
        which this source code may be used, no warranty of fitness for a 
        particular purpose is offered. The user is advised to test the source 
        code thoroughly before relying on it. The user must assume the entire 
        risk of using the source code.
    """
    for ix, f in enumerate(folders):
        if utilities:
            with open(f+'/U_dc.pkl', 'rb') as o:
                U_dc = pickle.load(o)
            with open(f+'/U_du.pkl', 'rb') as o:
                U_du = pickle.load(o)
            with open(f+'/U_ac.pkl', 'rb') as o:
                U_ac = pickle.load(o)
            with open(f+'/U_au.pkl', 'rb') as o:
                U_au = pickle.load(o)
            with open(f+'/loopList.pkl', 'rb') as o:
                loopList = pickle.load(o)
            with open(f+'/def_utility_umat_varies.pkl', 'rb') as o:
                resultsUncert = pickle.load(o)
            with open(f+'/def_utility_gamma_varies.pkl', 'rb') as o:
                resultsGamma = pickle.load(o)

            #combine these pickles with existing pickles
            if ix == 0:
                totalResultsUncert = copy.deepcopy(resultsUncert)
                totalResultsGamma = copy.deepcopy(resultsGamma)
            else:
                totalResultsUncert = np.vstack((totalResultsUncert, 
                                                resultsUncert))
                totalResultsGamma = np.vstack((totalResultsGamma, 
                                               resultsGamma))
        elif timing:
            with open(f+'/timing_results.pkl', 'rb') as o:
                resultsTiming = pickle.load(o)
            with open(f+'/minMaxInc.pkl', 'rb') as o:
                minMaxInc = pickle.load(o)
                if ix == 0:
                    minN = minMaxInc[0]
                    maxN = minMaxInc[1]
                    incN = minMaxInc[2]
                else:
                    if (minN != minMaxInc[0]) or (maxN != minMaxInc[1]) or \
                       (incN != minMaxInc[2]):
                        print('Error: Not all min, max, inc are same!')
                        sys.exit(1)

            #combine these pickles with existing pickles
            if ix == 0:
                totalResultsTiming = copy.deepcopy(resultsTiming)
            else:
                totalResultsTiming = np.vstack((totalResultsTiming, 
                                                resultsTiming))

        elif ratio:
            with open(f+'/ratio_results.pkl', 'rb') as o:
                resultsRatio = pickle.load(o)
            
            #combine these pickles with existing pickles
            if ix == 0:
                totalResultsRatio = copy.deepcopy(resultsRatio)
            else:
                totalResultsRatio = np.vstack((totalResultsRatio, 
                                                resultsRatio))


    if timing:
        #3a, 3b
        generate_heuristic_plot(totalResultsTiming, 
                                minN, 
                                maxN, 
                                incN, 
                                mode=mode,
                                save=save, 
                                show=show)
    elif utilities:
        #3c, 3d
        generate_defender_utilities_plot(totalResultsGamma, 
                                         totalResultsUncert, 
                                         U_dc, 
                                         U_du, 
                                         U_ac, 
                                         U_au,
                                         save=save,
                                         show=show)
        #3e
        generate_conditional_prob_plot(loopList, 
                                       totalResultsGamma, 
                                       save=save, 
                                       show=show)

    elif ratio:
        #3f, 3g
        generate_ratio_plots(totalResultsRatio, save=save, show=show)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Combine cluster results.')
    parser.add_argument('folders', nargs='+',
                        help='List folders containing final pickles, e.g., '+\
                             '"1" "2"...')
    parser.add_argument('-t', '--timing',
                        help='Include to run timing plots (3a, 3b).',
                        default=False,
                        action='store_true')
    parser.add_argument('-u', '--utilities',
                        help='Include to run utilities plots (3c, 3d, 3e).',
                        default=False,
                        action='store_true')
    parser.add_argument('-r', '--ratio',
                        help='Include to run ratio plots (3f, 3g).',
                        default=False,
                        action='store_true')
    parser.add_argument('-m', '--mode',
                        help='Plot 3a, 3b, or both based on desired range.',
                        default='3a')
    parser.add_argument('-s', '--save',
                        help='Include if you wish to save plots as files.',
                        default=False,
                        action='store_true')
    parser.add_argument('-nsh', '--show',
                        help='"No show": Include if you DO NOT wish to show '+\
                             'the plots.',
                        default=True,
                        action='store_false')
    args = parser.parse_args()
    plot_cluster_results(args.folders,
                         args.timing,
                         args.utilities,
                         args.ratio,
                         args.mode,
                         args.save,
                         args.show)