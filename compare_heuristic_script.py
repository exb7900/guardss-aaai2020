import networkx as nx
import matplotlib.pyplot
import numpy as np
import time
from relax import nonzeroColumnMethodRelaxation
from random_strategy import enumerateAll
from both_detection_fn_signaling_uncertainty \
    import nonzeroCplexSolverFixedTarget
from random_strategy import greedyInitialStrategy
from random_strategy import randomStrategy
from max_iteration import maxIterationComb
import argparse
import pickle
from scipy import stats

def compare_heuristic_script(minN,
                             maxN,
                             incN,
                             NUMBEROFGRAPHS,
                             method,
                             plot = True,
                             mode = '3a',
                             save = False,
                             show = True,
                             savePickle=False, 
                             loadPickle=False, 
                             randomSeed=False, 
                             graphType='watts'):
    """
    title::
        compare_heuristic_script

    description::
        This method will run timing experiments comparing the various versions
        of our AAAI-2020 paper, including multiple LPs method without branch
        and price vs. branch and price vs. branch and price plus warmup. Leads
        to Figs. 3a-3b in paper. 

    attributes::
        minN
            Minimum number of targets (N) in graph to time 
        maxN
            Maximum number of targets (N) in graph to time
        incN
            Increment between min and max targets in graph.
        NUMBEROFGRAPHS
            Number of graphs to time for each graph size N.
        method
            Which algorithm versions to compare:
                "all" to all of the below methods
                "lp" to run multiple LPs method without speedup
                "bp" to run branch and price
                "warmup" to run branch and price plus warmup
        plot
            Set to True if you wish to plot Fig. 3a, 3b [default is True]
        mode
            '3a', '3b', or 'both' for which plot to create based on the current
            range (for ease since they cover different ranges in the paper)
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is True] NOTE: do not set both save and show to True, one may be 
            blank
        savePickle
            If True, save all random graphs generated.
        loadPickle
            If True, load previously-generated graphs (will fail if no graph
            pickles exist).
        randomSeed
            If True, use same random seed during solution.
        graphType
            Which graph type to use for random generation:
                "watts" for Watts Strogatz graphs
                "cycle" for cycle graphs

    returns::
        resultsTime
            Depth: objective value exponential, time exponential, objective 
            value branch & price, time branch & price, objective value branch &
            price + warm-up, time branch & price + warm-up

            Rows: Results for graphs

            Columns: Results for different N's

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
    NS = (maxN-minN)//incN

    resultsTime = np.zeros((NUMBEROFGRAPHS, NS, 6))


    #Iterate over all graph sizes provided.
    for N in range(minN,maxN,incN):
        print('N', N)
        #For each size, run for number of random graphs requested.
        for i in range(NUMBEROFGRAPHS):
            print('i', i)
            k = max(int(np.sqrt(N)/2),1)
            l = int(((N*2)/3)-k)
            p = 0.1
            maxIteration = max(500, int(N*N/5))
            maxIteration = min(maxIteration,maxIterationComb(N,k,l))

            #NOTE: may want to add random seed for graph generation 
            #(or something else) if running many in parallel.
            if graphType == 'watts':
                G = nx.connected_watts_strogatz_graph(N, 2, p)
            else:
                G = nx.cycle_graph(N)
 
            #Generate random utilities.
            maximumPayment = 10
            U_dc = np.random.random(N) * maximumPayment
            U_dc += 0.9
            U_dc /= U_dc.max()
            U_du = np.random.random(N) * maximumPayment
            U_du = ((U_du+0.9) / U_du.max()) * -1000
            U_ac = np.random.random(N) * maximumPayment # for non-zero sum game
            U_ac = ((U_ac+0.9) / U_ac.max()) * -1
            U_au = np.random.random(N) * maximumPayment # for non zero sum game
            U_au = (U_au+0.9) * 2

            #Save/load utilities.
            if savePickle:
                with open('G_'+str(i)+str(N)+'.pkl','wb') as o:
                    pickle.dump(G, o)
                with open('U_dc'+str(i)+str(N)+'.pkl', \
                  'wb') as o:
                    pickle.dump(U_dc, o)
                with open('U_du' +str(i)+str(N)+'.pkl', \
                  'wb') as o:
                    pickle.dump(U_du, o)
                with open('U_ac' +str(i)+str(N)+'.pkl', \
                  'wb') as o:
                    pickle.dump(U_ac, o)
                with open('U_au' +str(i)+str(N)+'.pkl', \
                  'wb') as o:
                    pickle.dump(U_au, o)

            if loadPickle:
                with open('G_'+str(i)+str(N)+'.pkl','rb') as o:
                    G = pickle.load(o)
                with open('U_dc'+str(i)+str(N)+'.pkl','rb') as o:
                    U_dc = pickle.load(o)
                with open('U_du'+str(i)+str(N)+'.pkl','rb') as o:
                    U_du = pickle.load(o)
                with open('U_ac'+str(i)+str(N)+'.pkl','rb') as o:
                    U_ac = pickle.load(o)
                with open('U_au'+str(i)+str(N)+'.pkl','rb') as o:
                    U_au = pickle.load(o)

            #Fix detection and observational uncertainty including attacker
            #behavior vector.
            gamma = 0.5
            rnd = 0.5
            uncertaintyMatrix = np.array([[1,     rnd,     rnd/2],
                                          [0,     1-rnd,   rnd/2],
                                          [0,     0.00,    1-rnd]])
            
            eta = [[0,0,0],
                 [1,0,0],
                 [0,1,0],
                 [0,0,1],
                 [1,1,0],
                 [1,0,1],
                 [0,1,1],
                 [1,1,1]]

            #Run without speedups (solve LP for all targets and potential
            #behaviors), with about 1 hr time cutoff (not exact since
            #finishes solving first). Record best objective value throughout.
            if method == 'all' or method == 'lp':
                # No speedups
                if N < 20:
                    startTime = time.time()
                    AllStrategies = enumerateAll(G,k,l)
                    curOBJ = -float('inf')
                    bestA = None
                    bestTarget = None
                    halt = False
                    for TARGET in range(N):
                        for ai in eta:
                            resLP, variables, cpx = \
                                nonzeroCplexSolverFixedTarget(AllStrategies,
                                                  G,
                                                  N,
                                                  k,
                                                  l,
                                                  gamma,
                                                  ai,
                                                  uncertaintyMatrix,
                                                  U_dc,
                                                  U_du,
                                                  U_ac,
                                                  U_au,
                                                  TARGET)
                            if resLP > curOBJ:
                                curOBJ = resLP
                                bestA = ai
                                bestTarget = TARGET
                                bestVar = variables
                            if time.time() - startTime > 3600:
                                print(curOBJ)
                                halt = True
                                break
                        if halt:
                            break
  
                    StopTimeLP = time.time()-startTime
                    resultsTime[i,(N-minN)//incN,0] =  curOBJ
                    resultsTime[i,(N-minN)//incN,1] = StopTimeLP
          
            #Run branch and price without warmup.
            if method == 'all' or method == 'bp':
                # Branch and price
                startTime = time.time()
                #Get initial strategies for this pass.
                if 500 > maxIterationComb(N,k,l):
                    initialStrategies = enumerateAll(G,k,l,MAXIteration=500)
                else:
                    initialStrategies = []
                    for iS in range(500):
                        initialStrategies.append(randomStrategy(G, k, l))
                resBoth, variables, cpx, count, strategies, bestA, bestTarget=\
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
                StopTimeRel = time.time() - startTime
                resultsTime[i,(N-minN)//incN,2] =  resBoth
                resultsTime[i,(N-minN)//incN,3] = StopTimeRel
            

            #Run branch and price plus warm up.
            if method == 'all' or method == 'warmup':      
                # Branch and price + warm up
                startTime = time.time()
                #Get initial strategies for this pass.
                if 500 > maxIterationComb(N,k,l):
                    initialStrategies = enumerateAll(G,k,l,MAXIteration=500)
                else:
                    initialStrategies = \
                        greedyInitialStrategy(G,k,l,U_dc,U_du,U_ac,U_au)
                resBoth, variables, cpx, count, strategies, bestA, bestTarget=\
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
                StopTimeRelWar = time.time() - startTime
                resultsTime[i,(N-minN)//incN,4] =  resBoth
                resultsTime[i,(N-minN)//incN,5] = StopTimeRelWar


    #Writes results to pickle.
    with open('timing_results.pkl', 'wb') as o:
        pickle.dump(resultsTime, o)
    with open('minMaxInc.pkl', 'wb') as o:
        pickle.dump([minN, maxN, incN], o)

    if plot:
        generate_heuristic_plot(resultsTime, 
                                minN, 
                                maxN, 
                                incN, 
                                mode=mode,
                                save=save, 
                                show=show)

    return resultsTime

def generate_heuristic_plot(resultsTime, 
                            minN, 
                            maxN, 
                            incN, 
                            mode='3a',
                            save=False, 
                            show=False):
    """
    title::
        generate_heuristic_plot

    description::
        This method will plot Figs. 3a-3b in paper. 

    attributes::
        resultsTime
            Depth: objective value exponential, time exponential, objective 
            value branch & price, time branch & price, objective value branch &
            price + warm-up, time branch & price + warm-up

            Rows: Results for graphs

            Columns: Results for different N's
        minN
            Minimum number of targets (N) in graph to time 
        maxN
            Maximum number of targets (N) in graph to time
        incN
            Increment between min and max targets in graph.
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
    #Report ttests and number of instances exceeding time cutoff.
    print('ttest exp LP vs. b&p', stats.ttest_ind(resultsTime[:,:,1], 
                                                  resultsTime[:,:,3], 
                                                  axis=0))
    print('ttest exp LP vs. b&p+w', stats.ttest_ind(resultsTime[:,:,1], 
                                                    resultsTime[:,:,5], 
                                                    axis=0))
    print('ttest b&p vs. b&p+w', stats.ttest_ind(resultsTime[:,:,3], 
                                                 resultsTime[:,:,5], 
                                                 axis=0))
    print('number > 3600 (exp): ', np.where(resultsTime[:,:,1] \
                                                      >= 3600)[1].shape[0])
    print('number > 3600 (b&p): ', np.where(resultsTime[:,:,3] \
                                                      >= 3600)[1].shape[0])
    print('number > 3600 (b&p+w): ', np.where(resultsTime[:,:,5] \
                                                        >= 3600)[1].shape[0])


    N = np.arange(minN, maxN, incN)
    resultsTimeMean = np.mean(resultsTime, axis=0)
    resultsTimeMean = np.clip(resultsTimeMean, 0, 3600)

    if (mode == '3a') or (mode == 'both'):
        matplotlib.pyplot.rcParams.update({'font.size': 22})
        matplotlib.pyplot.figure()
        matplotlib.pyplot.xlabel('Number of Targets')
        matplotlib.pyplot.ylabel('Runtime (s)')
        matplotlib.pyplot.xticks(np.arange(6, 18, step=2))
        matplotlib.pyplot.yticks(np.arange(0, 4800, step=1200))
        matplotlib.pyplot.hlines(3600, 6, 16, colors='r', linestyles='dashed')
        matplotlib.pyplot.plot(N, resultsTimeMean[:,1],
                                    label='Exponential LP')
        matplotlib.pyplot.plot(N, resultsTimeMean[:,3],
                                    '--',color='C1',label='Branch+Price')
        matplotlib.pyplot.plot(N, resultsTimeMean[:,5],
                                    's-',color='C2',label='Branch+Price+Warm-up')
        matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                                borderaxespad=0., mode='expand')
        if save:
            matplotlib.pyplot.savefig('timing_3a.pdf', bbox_inches='tight')


    if (mode == '3b') or (mode == 'both'):
        matplotlib.pyplot.rcParams.update({'font.size': 22})
        matplotlib.pyplot.figure()
        matplotlib.pyplot.xlabel('Number of Targets')
        matplotlib.pyplot.ylabel('Runtime (s)')
        matplotlib.pyplot.xticks(np.arange(20, 120, step=20))
        matplotlib.pyplot.yticks(np.arange(0, 4800, step=1200))
        matplotlib.pyplot.hlines(3600, 20, 100, colors='r', linestyles='dashed')
        matplotlib.pyplot.plot(N, resultsTimeMean[:,3],
                                    '--',color='C1',label='Branch+Price')
        matplotlib.pyplot.plot(N, resultsTimeMean[:,5],
                                    's-',color='C2',label='Branch+Price+Warm-up')
        matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                                borderaxespad=0., mode='expand')
        if save:
            matplotlib.pyplot.savefig('timing_3b.pdf', bbox_inches='tight')
    if show:
        matplotlib.pyplot.show()

                                               

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run timing experiments.')
    parser.add_argument('minN', type=int,
                        help='Minimum number of nodes (N) in graph to time.')
    parser.add_argument('maxN', type=int,
                        help='Maximum number of nodes (N) in graph to time.')
    parser.add_argument('incN', type=int,
                        help='Increment between min and max nodes in graph.')
    parser.add_argument('NUMBEROFGRAPHS', type=int,
                        help='Number of graphs to time for each graph size N.')
    parser.add_argument('method',
                        help='"all" to all of the below methods;'+
                           '"lp" to run multiple LPs method without speedup;'+
                           '"bp" to run branch and price;'+
                           '"warmup" to run branch and price plus warmup.')
    parser.add_argument('-np', '--plot',
                        help='"No plot": Include if you DO NOT wish to plot '+\
                             'Fig. 3a,b.',
                        default=True,
                        action='store_false')
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
    parser.add_argument('-sp', '--savePickle',
                        help='Include to save all random graphs generated.',
                        default=False,
                        action='store_true')
    parser.add_argument('-lp', '--loadPickle',
                        help='Include to load previously-generated graphs.',
                        default=False,
                        action='store_true')
    parser.add_argument('-r', '--randomSeed',
                        help='Include to use same random seed during solution.',
                        default=False,
                        action='store_true')
    parser.add_argument('-cg', '--cycleGraphOn',
                        help='Include to use cycle graphs instead.',
                        default='watts',
                        action='store_const',
                        const='cycle')
    args = parser.parse_args()
    resultsTime = compare_heuristic_script(args.minN,
                                           args.maxN,
                                           args.incN,
                                           args.NUMBEROFGRAPHS,
                                           args.method,
                                           plot=args.plot,
                                           mode=args.mode,
                                           save=args.save,
                                           show=args.show,
                                           savePickle=args.savePickle,
                                           loadPickle=args.loadPickle,
                                           randomSeed=args.randomSeed,
                                           graphType=args.cycleGraphOn)
