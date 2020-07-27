import networkx as nx
import matplotlib.pyplot
import numpy as np
from relax import nonzeroColumnMethodRelaxation
from random_strategy import greedyInitialStrategy
from random_strategy import randomStrategy
from random_strategy import enumerateAll
from max_iteration import maxIterationComb
from scipy import stats
import argparse

MUTE = 1

def number_attackers_uavs_over_gamma_script(N,
                                            MAXK=4,
                                            plot=True,
                                            NUMBEROFGRAPH=20,
                                            savePickle=False, 
                                            loadPickle=False,
                                            loadGraph=False, 
                                            saveGraph=False,
                                            randomSeed=False, 
                                            warmUp=True,
                                            save=False,
                                            show=True,
                                            saveResults=False):
    """
    title::
        number_attackers_uavs_over_gamma_script

    description::
        This method will run experiments leading to Figs. 3f-3g in the paper. 

    attributes::
        N
            Number of targets (graph size) (15 used in paper)
        MAXK
            Maximum number of human patrollers to test [default is 4 (paper)]
        plot
            Set to True if you wish to plot Fig. 3c, 3d, 3e [default is True]
        NUMBEROFGRAPH
            Total number of graphs to use for averaging [default is 20 (in 
            paper)]
        savePickle
            If True, save all random utilities generated [default is False]
        loadPickle
            If True, load previously-generated utilities
            (will fail if no pickles exist) [default is False]
        loadGraph
            If True, load previously-generated graphs
            (will fail if no graph pickles exist) [default is False]
        saveGraph
            If True, save all random graphs [default is False]
        randomSeed
            If True, use same random seed during solution [default is False]
        warmUp
            Whether to use the greedy warm-up strategy proposed to generate
            initial strategies [default is False]
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is True] NOTE: do not set both save and show to True, one may be 
            blank
        saveResults
            Whether to save the final results arrays as pickles or not [default
            is False]

    returns::
        resultsRatio
            Depth: objective value k=MAXK, ratio=0,
                   objective value k=MAXK-1, ratio=1,
                   objective value k=MAXK-1, ratio=2,
                   objective value k=MAXK-1, ratio=3

            Rows: Results for graphs

            Columns: Results for gammas

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
    #Initialize uncertainty matrix, eta, results, utilities.
    resultsRatio = np.zeros((NUMBEROFGRAPH, 10, 4))
    eta = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
    gamma = 0.0
    uncertaintyMatrix = np.array([[1,     0,     0],
                                  [0,     1,     0],
                                  [0,     0.00,    1]])

    maximumPayment = 10
    U_dc = np.random.random(N)*maximumPayment
    U_dc += 0.9
    U_dc /= U_dc.max()
    U_du = np.random.random(N)*maximumPayment
    U_du = ((U_du+0.9) / U_du.max()) * -1000
    U_ac = np.random.random(N)*maximumPayment
    U_ac = ((U_ac+0.9) / U_ac.max()) * -1
    U_au = np.random.random(N)*maximumPayment
    U_au = (U_au+0.9) * 2

    if savePickle:
        import pickle
        with open('U_dc.pkl','wb') as o:
            pickle.dump(U_dc, o)
        with open('U_du.pkl','wb') as o:
            pickle.dump(U_du, o)
        with open('U_ac.pkl','wb') as o:
            pickle.dump(U_ac, o)
        with open('U_au.pkl','wb') as o:
            pickle.dump(U_au, o)
    if loadPickle:
        import pickle
        with open('U_dc.pkl','rb') as o:
            U_dc = pickle.load(o)
        with open('U_du.pkl','rb') as o:
            U_du = pickle.load(o)
        with open('U_ac.pkl','rb') as o:
            U_ac = pickle.load(o)
        with open('U_au.pkl','rb') as o:
            U_au = pickle.load(o)
    
    #Initialize the two k's (# patrollers) that will be tested.
    KList = [MAXK,MAXK-1]

    #Loop through number of graphs.
    for i in range(NUMBEROFGRAPH):
        if MUTE == 1: print('graph', i)
        #Start counter at 0 for resultsRatio depth.
        ratioIndex = 0

        #Initialize this graph.
        G = nx.connected_watts_strogatz_graph(N, 3, 0.7)
        if saveGraph:
            with open('G_' + str(i) + \
                      '.pkl','wb') as o:
                pickle.dump(G, o)
        if loadGraph:
            with open('G_' + str(i) + \
                      '.pkl','rb') as o:
                G = pickle.load(o)
                
        #Loop through k's.
        for ik in range(len(KList)):
            #Only do 0 if k=MAXK, only do [1,2,3] if k=MAXK-1
            if ik == 0:
                ratios = [0]
            else:
                ratios = [1,2,3]

            #Loop through [0] or [1,2,3].
            for ratio in ratios:
                if MUTE == 1: print('ratio', ratio)
                #Fix k and l.
                k = KList[ik]
                l = min(N-k, (MAXK-k)*ratio)
                maxIteration = max(500, int(N*N/5))
                maxIteration = min(maxIteration,maxIterationComb(N,k,l))

                #Loop through gammas and run.
                if l > 0: #(don't need to do different gammas if no drones!)
                    for INDEX, gamma in enumerate(np.arange(0,1,0.1)):
                        if MUTE == 1: print('gamma', gamma)
                        #Get initial strategies for this pass.
                        if 500 > maxIterationComb(N,k,l):
                            initialStrategies = enumerateAll(G,
                                                            k,
                                                            l,
                                                            MAXIteration=500)
                        elif warmUp == True:
                            initialStrategies = greedyInitialStrategy(G,
                                                                    k,
                                                                    l,
                                                                    U_dc,
                                                                    U_du,
                                                                    U_ac,
                                                                    U_au)
                        else:
                            initialStrategies = []
                            for iS in range(500):
                                initialStrategies.append(randomStrategy(G, k, l))

                        #Run with uncertainty.
                        resBoth, variables, cpx, count, strategies, bestEtaBoth, \
                            bestTargetBoth = \
                                nonzeroColumnMethodRelaxation(N,
                                                            k,
                                                            l,
                                                            G,
                                                            gamma,
                                                            eta,
                                                            uncertaintyMatrix,
                                                            U_dc,
                                                            U_du,
                                                            U_ac,
                                                            U_au,
                                                            initialStrategies,
                                                            randomSeed=randomSeed,
                                                            maxIteration=maxIteration)
                            
                        resultsRatio[i,INDEX,ratioIndex] = resBoth
                        
                    
                else:
                    #Get initial strategies for this pass.
                    if 500 > maxIterationComb(N,k,l):
                        initialStrategies = enumerateAll(G,
                                                        k,
                                                        l,
                                                        MAXIteration=500)
                    elif warmUp == True:
                        initialStrategies = greedyInitialStrategy(G,
                                                                k,
                                                                l,
                                                                U_dc,
                                                                U_du,
                                                                U_ac,
                                                                U_au)
                    else:
                        initialStrategies = []
                        for iS in range(500):
                            initialStrategies.append(randomStrategy(G, k, l))

                    #Run with uncertainty.
                    resBoth, variables, cpx, count, strategies, bestEtaBoth, \
                        bestTargetBoth = \
                            nonzeroColumnMethodRelaxation(N,
                                                        k,
                                                        l,
                                                        G,
                                                        gamma,
                                                        eta,
                                                        uncertaintyMatrix,
                                                        U_dc,
                                                        U_du,
                                                        U_ac,
                                                        U_au,
                                                        initialStrategies,
                                                        randomSeed=randomSeed,
                                                        maxIteration=maxIteration)
                    resultsRatio[i,:,ratioIndex] = resBoth

                ratioIndex += 1

    #Writes results to pickle.
    if saveResults:
        with open('ratio_results.pkl', 'wb') as o:
            pickle.dump(resultsRatio, o)

    if plot:
        generate_ratio_plots(resultsRatio, save=save, show=show)


    


def generate_ratio_plots(resultsRatio, save=False, show=True):
    """
    title::
        generate_ratio_plots

    description::
        This method will plot Figs. 3f-3g and print statistics in the paper.

    attributes:
        resultsRatio
            Depth: objective value k=MAXK, ratio=0,
                   objective value k=MAXK-1, ratio=1,
                   objective value k=MAXK-1, ratio=2,
                   objective value k=MAXK-1, ratio=3

            Rows: Results for graphs

            Columns: Results for gammas
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
    print('ttest 0,1',stats.ttest_ind(resultsRatio[:,:,0], 
                                      resultsRatio[:,:,1], 
                                      axis=0))
    print('ttest 0,2',stats.ttest_ind(resultsRatio[:,:,0], 
                                      resultsRatio[:,:,2], 
                                      axis=0))
    print('ttest 0,3',stats.ttest_ind(resultsRatio[:,:,0], 
                                      resultsRatio[:,:,3], 
                                      axis=0))
    print('ttest 1,2',stats.ttest_ind(resultsRatio[:,:,1], 
                                      resultsRatio[:,:,2], 
                                      axis=0))
    print('ttest 2,3',stats.ttest_ind(resultsRatio[:,:,2], 
                                      resultsRatio[:,:,3], 
                                      axis=0))
    print('ttest 1,3',stats.ttest_ind(resultsRatio[:,:,1], 
                                      resultsRatio[:,:,3], 
                                      axis=0))

    resultsRatioMean = np.mean(resultsRatio, axis=0)
    resultsRatioStd = np.std(resultsRatio, axis=0)

    #3f
    matplotlib.pyplot.rcParams.update({'font.size':22})
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xlabel('Amount of Detection Uncertainty ($\gamma$)')
    matplotlib.pyplot.ylabel("Defender's Expected Utility")
    matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsRatioMean[:,1], '-',
                        label='k=3, l=1')
    matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsRatioMean[:,2], '^--',
                        label='k=3, l=2')
    matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsRatioMean[:,3], 'p-.',
                        label='k=3, l=3')
    matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
                             loc=3, 
                             ncol=1, 
                             mode="expand", 
                             borderaxespad=0.)
    if save:
        matplotlib.pyplot.savefig('ratio_plot.pdf',bbox_inches='tight')

    #3g
    matplotlib.pyplot.rcParams.update({'font.size':22})
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xlabel('Amount of Detection Uncertainty ($\gamma$)')
    matplotlib.pyplot.ylabel("Defender's Expected Utility")
    matplotlib.pyplot.bar([-3,2,7], 
                          [resultsRatioMean[3,0],
                           resultsRatioMean[5,0],
                           resultsRatioMean[8,0]],
                          [1,1,1], 
                          color=(0.3, 0.3, 0.3, 1.0), 
                          label="no drones (k=4, l=0)", 
                          yerr=[resultsRatioStd[3,0],
                                resultsRatioStd[5,0],
                                resultsRatioStd[8,0]])
    matplotlib.pyplot.bar([-2,3,8], 
                          [resultsRatioMean[3,2],
                           resultsRatioMean[5,2],
                           resultsRatioMean[8,2]],
                          [1,1,1],
                          tick_label = [0.3,0.5,0.8], 
                          label="k=3, l=2", 
                          color='C1', 
                          hatch = "/", 
                          yerr=[resultsRatioStd[3,2],
                                resultsRatioStd[5,2],
                                resultsRatioStd[8,2]])
    matplotlib.pyplot.bar([-1,4,9], 
                          [resultsRatioMean[3,3],
                           resultsRatioMean[5,3],
                           resultsRatioMean[8,3]],
                          [1,1,1], 
                          label = "k=3, l=3", 
                          color='C2', 
                          hatch ="+", 
                          yerr=[resultsRatioStd[3,3],
                                resultsRatioStd[5,3],
                                resultsRatioStd[8,3]])
    matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
                             loc=3, 
                             ncol=1, 
                             mode="expand", 
                             borderaxespad=0.)
    if save:
        matplotlib.pyplot.savefig('ratio_bar.pdf',bbox_inches='tight')
    if show:
        matplotlib.pyplot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ratio plot '+\
                                                 'experiments for 3f, 3g.')
    parser.add_argument('N', type=int,
                        help='Number of targets (graph size).')
    parser.add_argument('-k', '--MAXK', type=int,
                        default=4,
                        help='Maximum number of human patrollers to test.')
    parser.add_argument('-np', '--plot',
                        help='"No plot": Include if you DO NOT wish to plot '+\
                             'Fig. 3c,d,e.',
                        default=True,
                        action='store_false')                    
    parser.add_argument('-s', '--save',
                        help='Include if you wish to save plots as files.',
                        default=False,
                        action='store_true')
    parser.add_argument('-nsh', '--show',
                        help='"No show": Include if you DO NOT wish to show '+\
                             'the plots.',
                        default=True,
                        action='store_false')
    parser.add_argument('-sp', '--savePickle',
                        help='Include to save graphs, utilities.',
                        default=False,
                        action='store_true')
    parser.add_argument('-lp', '--loadPickle',
                        help='Include to load utilities.',
                        default=False,
                        action='store_true')
    parser.add_argument('-lg', '--loadGraph',
                        help='Include to load graphs.',
                        default=False,
                        action='store_true')
    parser.add_argument('-sg', '--saveGraph',
                        help='Include to save graphs.',
                        default=False,
                        action='store_true')
    parser.add_argument('-r', '--randomSeed',
                        help='Include to use same random seed during solution.',
                        default=False,
                        action='store_true')
    parser.add_argument('-w', '--warmUp',
                        help='Include to use greedy warm-up strategy.',
                        default=False,
                        action='store_true')
    parser.add_argument('-g', '--NUMBEROFGRAPH', 
                        help='Number of graphs to test.',
                        type=int,
                        default=20)
    parser.add_argument('-sr', '--saveResults',
                        help='Include to save the final result arrays as '+\
                             'pickles.',
                        default=False,
                        action='store_true')
    
    args = parser.parse_args()

    number_attackers_uavs_over_gamma_script(args.N,
                                            args.MAXK,
                                            NUMBEROFGRAPH=args.NUMBEROFGRAPH,
                                            plot=args.plot,
                                            savePickle=args.savePickle, 
                                            loadPickle=args.loadPickle, 
                                            loadGraph=args.loadGraph,
                                            saveGraph=args.saveGraph,
                                            randomSeed=args.randomSeed, 
                                            warmUp=args.warmUp,
                                            save=args.save,
                                            show=args.show,
                                            saveResults=args.saveResults)