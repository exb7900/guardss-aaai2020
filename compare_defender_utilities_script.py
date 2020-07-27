import networkx as nx
import matplotlib.pyplot
import numpy as np
import pickle
from relax import nonzeroColumnMethodRelaxation
from random_strategy import greedyInitialStrategy
from random_strategy import randomStrategy
from random_strategy import enumerateAll
from max_iteration import maxIterationComb
from compute_real_optimal_attacker import compute_real_optimal_attacker
import argparse
import copy
from scipy import stats

MUTE = 1

def compare_defender_utilities_script(N,
                                      k,
                                      l,
                                      plot = True,
                                      save = False,
                                      show = True,
                                      annotateGraph = False,
                                      saveVariables = True,
                                      robustnessTest = False,
                                      testGamma = True,
                                      testUMat = True,
                                      savePickle = False,
                                      loadPickle = False,
                                      loadGraph=False, 
                                      saveGraph=False,
                                      randomSeed = False, 
                                      warmUp = False,
                                      runWithOneUncertainty = True,
                                      graphType = 'watts',
                                      NUMBEROFGRAPH = 20,
                                      runOneGammaAndUMat = False,
                                      saveResults = False):

    """
    title::
        compare_defender_utilities_script

    description::
        This method will run experiments leading to Figs. 3c-3e in the paper. 
        This method will also generate statistics presented in paper Section 6,
        and ttest results. NOTE: All savings are False by default in order to
        prevent accidentally overwriting something (which is never checked 
        before writing). Please be careful not to accidentally overwrite. Also,
        most of the settings are only included for detection uncertainty, as it
        is easier to test and interpret without the attacker behaviors.

    attributes::
        N
            Number of targets (graph size)
        k
            Number of human patrollers
        l
            Number of drones
        plot
            Set to True if you wish to plot Fig. 3c, 3d, 3e [default is True]
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is True] NOTE: do not set both save and show to True, one may be 
            blank
        annotateGraph
            Set to True if you wish to show graphs with annotations of 
            utilities, decision variables after solution [default is False]
        saveVariables
            Set to True if you wish to calculate the statistics used in Section
            6 in the paper (only runs for gamma = 0.5 as in paper) [default is 
            True]
        robustnessTest
            Set to True if you wish to test an assumed *gamma* (0.7) that is 
            close to the true gamma (0.8) [default is False] NOTE: initial
            tests from paper was primarily done in case_study.py, but this 
            was added for convenience to scale up testing in the future 
            (from running a few times here, it showed lots of variation with
            different utilities and graphs) (also, using percent difference
            here, but percent change was used in the paper - the change was
            due to the ambiguity in choosing the first and last numbers)
        testGamma
            Set to True if you wish to test detection uncertainty (i.e., 
            fix the uncertainty matrix, increment gamma, which would ultimately
            lead to Figure 3c) [default is True]
        testUMat
            Set to True if you wish to test observational uncertainty (i.e., 
            fix gamma, increment uncertainty matrix, which would ultimately 
            lead to Figure 3d) [default is True]
        savePickle
            If True, save all random utilities generated [default is False]
        loadPickle
            If True, load previously-generated utilities
            (will fail if no graph pickles exist) [default is False]
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
        runWithOneUncertainty
            When incrementing over uncertainties, use no other uncertainty at
            the same time (i.e., when incrementing gamma and fixing the 
            uncertainty matrix, kappa=0, uMat=I) [default is True (in paper)]
        graphType
            Which graph type to use for random generation:
                "watts" for Watts Strogatz graphs (in paper) [default]
                "cycle" for cycle graphs
        NUMBEROFGRAPH
            Total number of graphs to use for averaging [default is 20 (in 
            paper)]
        runOneGammaAndUMat
            Set to True if you wish to run only 1 gamma and/or uncertainty 
            matrix for NUMBEROFGRAPH, so no plot or saveVariable reports if 
            True [default is False]
        saveResults
            Whether to save the final results arrays as pickles or not [default
            is False]
    
    returns::
        resultsGamma
            Abbreviations: objective value = OV, recalculated (i.e., computed
            as though no uncertainty, recalculated with true uncertainty) = 
            recalc, original (i.e., computed as though no uncertainty) = orig,
            target (i.e., attacker best response target) = tgt, eta (i.e., 
            attacker best response behavior) = eta

            Depth: both OV, recalc OV, orig OV, both tgt, recalc tgt, orig 
            tgt, both eta, both recalc, both orig, CONDITIONAL PROBABILITY MEAN

            Rows: Results for graphs

            Columns: Results for gammas

        resultsUncert
            Same as above, but no conditional probability mean (therefore, only 9 
            columns), and kappa instead of gamma

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
    #Uncertainty levels to test for gamma and uMat.
    if robustnessTest:
        runOneGammaAndUMat = True

    if runOneGammaAndUMat:
        loopList = np.array([0.3])
        if robustnessTest:
            loopList = np.array([0.8])

    else:
        loopList = np.arange(0,1,0.1)

    #Initialize results, eta, and utilities (since N does not change).
    resultsGamma = np.zeros((NUMBEROFGRAPH, loopList.shape[0], 10)) 
    resultsUncert = np.zeros((NUMBEROFGRAPH, loopList.shape[0], 9))

    maxIteration = max(500, int(N*N/5))
    maxIteration = min(maxIteration,maxIterationComb(N,k,l))

    maximumPayment = 10
    U_dc = np.random.random(N) * maximumPayment
    U_dc += 0.9
    U_dc /= U_dc.max()
    U_du = np.random.random(N) * maximumPayment
    U_du = ((U_du+0.9) / U_du.max()) * -1000
    U_ac = np.random.random(N) * maximumPayment
    U_ac = ((U_ac+0.9) / U_ac.max()) * -1
    U_au = np.random.random(N) * maximumPayment
    U_au = (U_au+0.9) * 2

    eta =[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

    if saveVariables:
        splusNo, nplusNo, ppNo, nminusNo, sbarNo, sminusNo = [], [], [], [], [], []
        splus, nplus, pp, nminus, sbar, sminus = [], [], [], [], [], []


    if savePickle:
        with open('U_dc.pkl', 'wb') as o:
            pickle.dump(U_dc, o)
        with open('U_du.pkl', 'wb') as o:
            pickle.dump(U_du, o)
        with open('U_ac.pkl', 'wb') as o:
            pickle.dump(U_ac, o)
        with open('U_au.pkl', 'wb') as o:
            pickle.dump(U_au, o)
        with open('loopList.pkl', 'wb') as o:
            pickle.dump(loopList, o)

    try:
        if loadPickle:
            with open('U_dc.pkl', 'rb') as o:
                U_dc = pickle.load(o)
            with open('U_du.pkl', 'rb') as o:
                U_du = pickle.load(o)
            with open('U_ac.pkl', 'rb') as o:
                U_ac = pickle.load(o)
            with open('U_au.pkl', 'rb') as o:
                U_au = pickle.load(o)
    except:
        print('WARNING: Failed to load utility pickles. Continuing with '+\
                'randomly generated utilities.')

    

    #Loop through graphs.
    for i in range(NUMBEROFGRAPH):
        if MUTE == 1: print('graph',i)
        #Initialize this graph.
        if graphType == 'cycle':
            G = nx.cycle_graph(N)
        else:
            G = nx.connected_watts_strogatz_graph(N, 3, 0.7)
        if annotateGraph:
            G1 = copy.deepcopy(G)
            G2 = copy.deepcopy(G)

        #Get initial strategies for this graph.
        if 500 > maxIterationComb(N,k,l):
            initialStrategies = enumerateAll(G,k,l, MAXIteration=500)
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

        if saveGraph:
            with open('G_' + str(i) + '.pkl', 'wb') as o:
                pickle.dump(G, o)
        try:
            if loadGraph:
                with open('G_' + str(i) + '.pkl', 'rb') as o:
                    G = pickle.load(o)

        except:
            print('WARNING: Failed to load graph pickles. '+\
                'Continuing with generated graph.')


        ############FIX GAMMA, INCREMENT UNCERTAINTY MAT#######################
        if testUMat:
            #Initialize uncertainties. Assumed uncertainties are for ignoring 
            #uncertainty.
            assumedGamma = 0
            assumedUncertaintyMatrix = np.array([[1.0,   0,   0],
                                                 [0,   1,   0],
                                                 [0,   0,   1]])

            #Run with ignored uncertainty.
            res, noVar, cpx, count, noStrategies, bestA, bestTarget = \
                        nonzeroColumnMethodRelaxation(N,
                                                    k,
                                                    l,
                                                    G,
                                                    assumedGamma,
                                                    eta,
                                                    assumedUncertaintyMatrix,
                                                    U_dc,
                                                    U_du,
                                                    U_ac,
                                                    U_au,
                                                    initialStrategies,
                                                    randomSeed=randomSeed,
                                                    maxIteration=maxIteration)
            
            #Gamma is fixed here, but could set to 0 or uncertainty for tests.
            if runWithOneUncertainty:
                gamma = 0
            else:
                gamma = 0.3
                              
            #Loop through kappas (uncertainty matrices).
            for INDEX, kappa in enumerate(loopList):
                if MUTE == 1: print('kappa', kappa)
                uncertaintyMatrix = np.array([[1,     kappa,       kappa/2],
                                              [0,     1-kappa,     kappa/2],
                                              [0,     0.00,        1-kappa]])

                #Run with both uncertainties.
                resBoth, variables, cpx, count, strategies, bestABoth, \
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
                resultsUncert[i,INDEX,0] = resBoth
                resultsUncert[i,INDEX,3] = bestTargetBoth
                resultsUncert[i,INDEX,6] = int(''.join(str(bestABoth) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(', ')))
            
                #Recalculate the ignored uncertainty for this level.
                realObj, btgt, ba = compute_real_optimal_attacker(N,
                                                             noVar,
                                                             noStrategies,
                                                             G,
                                                             uncertaintyMatrix,
                                                             gamma,
                                                             U_dc,
                                                             U_du,
                                                             U_ac,
                                                             U_au)

                resultsUncert[i,INDEX,2] = res
                resultsUncert[i,INDEX,1] = realObj
                resultsUncert[i,INDEX,4] = btgt
                resultsUncert[i,INDEX,7] = int(''.join(str(ba) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(' ')))
                resultsUncert[i,INDEX,5] = bestTarget
                resultsUncert[i,INDEX,8] = int(''.join(str(bestA) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(', ')))


            
        #############FIX UNCERTAINTY MAT, INCREMENT GAMMA######################
        if testGamma:
            #Initialize uncertainties. Assumed uncertainties are for ignoring 
            #uncertainty.
            if robustnessTest:
                assumedGamma = 0.7
            else:
                assumedGamma = 0
            assumedUncertaintyMatrix = np.array([[1.0,   0,   0],
                                                  [0,   1,   0],
                                                  [0,   0,   1]])

            #Run with ignored uncertainty.
            res, noVar2, cpx, count, noStrategies2, bestA, bestTarget=\
                        nonzeroColumnMethodRelaxation(N,
                                                    k,
                                                    l,
                                                    G,
                                                    assumedGamma,
                                                    eta,
                                                    assumedUncertaintyMatrix,
                                                    U_dc,
                                                    U_du,
                                                    U_ac,
                                                    U_au,
                                                    initialStrategies,
                                                    randomSeed=randomSeed,
                                                    maxIteration=maxIteration)

            #If wanting to annotate a graph with variable values for 
            #visualization.
            if annotateGraph:
                G2 = annotate_graph(noVar2, G2, N, U_dc, U_du, U_ac, U_au)

            #Kappa is fixed here, but could set to 0 or uncertainty for tests.
            if runWithOneUncertainty:
                kappa = 0
            else:
                kappa = 0.3
            uncertaintyMatrix = np.array([[1,     kappa,       0],
                                          [0,     1-kappa,     0],
                                          [0,     0.00,      1]])
            
            #Loop through gammas.
            for INDEX, gamma in enumerate(loopList):
                if MUTE == 1: print('gamma', gamma)
                #Run with both uncertainties.
                resBoth, variables, cpx, count, strategies, bestABoth, \
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
                resultsGamma[i,INDEX,0] = resBoth
                resultsGamma[i,INDEX,3] = bestTargetBoth
                resultsGamma[i,INDEX,6] = int(''.join(str(bestABoth) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(', ')))

                #Recalculate the ignored uncertainty for this level.
                realObj, btgt, ba = compute_real_optimal_attacker(N,
                                            noVar2,
                                            noStrategies2,
                                            G,
                                            uncertaintyMatrix, 
                                            gamma,
                                            U_dc,
                                            U_du,
                                            U_ac,
                                            U_au)
    
                resultsGamma[i,INDEX,2] = res
                resultsGamma[i,INDEX,1] = realObj
                resultsGamma[i,INDEX,4] = btgt
                resultsGamma[i,INDEX,7] = int(''.join(str(ba) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(' ')))
                resultsGamma[i,INDEX,5] = bestTarget
                resultsGamma[i,INDEX,8] = int(''.join(str(bestA) \
                                                    .split('[')[1] \
                                                    .split(']')[0] \
                                                    .split(', ')))

                if saveVariables: 
                    #At best target (for statistics).
                    if gamma == 0.5:
                        splusNo.append(variables[(5*N)+btgt]) #s+
                        nplusNo.append(variables[(1*N)+btgt]) #n+
                        ppNo.append(variables[btgt]) #p
                        nminusNo.append(variables[(2*N)+btgt]) #n-
                        sbarNo.append(variables[(3*N)+btgt]) #sbar
                        sminusNo.append(variables[(4*N)+btgt]) #s-

                        splus.append(variables[(5*N)+bestTargetBoth]) #s+
                        nplus.append(variables[(1*N)+bestTargetBoth]) #n+
                        pp.append(variables[bestTargetBoth]) #p
                        nminus.append(variables[(2*N)+bestTargetBoth]) #n-
                        sbar.append(variables[(3*N)+bestTargetBoth]) #sbar
                        sminus.append(variables[(4*N)+bestTargetBoth]) #s-  

                #For each target, calculate conditional prob in Fig. 3e/
                #Appendix C.
                xVar = np.array(variables[:6*N]).reshape((-1,N))
                psiVar = np.array(variables[6*N:9*N]).reshape((-1,N))
                phiVar = np.array(variables[9*N:12*N]).reshape((-1,N))
                sensorTgtsCP = []
                for t in range(N):
                    probSensor = xVar[3,t] + xVar[4,t] + xVar[5,t]
                    if probSensor > 0: #only calculate if there is a sensor
                        cp = calculate_conditional_prob(xVar, 
                                                        phiVar, 
                                                        psiVar, 
                                                        gamma, 
                                                        t)
                        sensorTgtsCP.append(cp)
                #Mean over all targets w/ sensors for this graph/gamma.
                #NOTE: Could also add mean psi/phi here if desired.
                currMean = np.mean(sensorTgtsCP)
                resultsGamma[i, INDEX, 9] = currMean

                #If wanting to annotate a graph with variable values for 
                #visualization (added 1 gamma to have fewer plots).
                if annotateGraph and (gamma == 0.3):
                    G1 = annotate_graph(variables, 
                                        G1, 
                                        N, 
                                        U_dc, 
                                        U_du, 
                                        U_ac, 
                                        U_au)
                    display_annotated_graph(G, G1, G2)  

    

    if saveResults:
        with open('def_utility_umat_varies.pkl', 'wb') as o:
            pickle.dump(resultsUncert, o)
        with open('def_utility_gamma_varies.pkl', 'wb') as o:
            pickle.dump(resultsGamma, o)

    #If wanting to compute statistics (in paper Section 6).
    if saveVariables and not runOneGammaAndUMat:
        no = np.vstack((splusNo, nplusNo, ppNo, nminusNo, sbarNo, sminusNo))
        uncert = np.vstack((splus, nplus, pp, nminus, sbar, sminus))
        calculate_statistics(no, uncert)

    if plot and not runOneGammaAndUMat:
        #3c and 3d
        generate_defender_utilities_plot(resultsGamma, 
                                         resultsUncert, 
                                         U_dc, 
                                         U_du, 
                                         U_ac, 
                                         U_au,
                                         save=save,
                                         show=show)
        #3e
        generate_conditional_prob_plot(loopList, 
                                       resultsGamma, 
                                       save=save, 
                                       show=show)
    if runOneGammaAndUMat:
        print('Actual result, graph 1, detection: ', resultsGamma[0,0,0])
        print('Assumed detection uncertainty, graph 1: ', resultsGamma[0,0,1])
        if robustnessTest:
            resultsGammaSqueezed = np.squeeze(resultsGamma[:,0,:2]).transpose()

            toMean = calculate_percent_difference(resultsGammaSqueezed)
            print('Mean difference over graphs: ', np.mean(toMean))

    return resultsGamma, resultsUncert

def annotate_graph(variables, G, N, U_dc, U_du, U_ac, U_au):
    """
    title::
        annotate_graph

    description::
        This method will add annotations to the graph object based on variables
        and utilities.

    attributes::
        variables
            Final optimal decision variable values
        G
            Graph object (networkx)
        N
            Number of targets (graph size)
        U_dc
            U_+^d (defender utility when defender successfully protects target)
        U_du
            U_-^d (defender utility when defender fails to protect target)
        U_ac
            U_+^a (attacker utility when defender successfully protects target)
        U_au
            U_-^a (attacker utility when defender fails to protect target)
    
    returns::
        G
            Graph object with annotations included

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
    xList = ['xp', 'xn+', 'xn-', 'xsbar', 'xs-', 'xs+']
    psiList = ['psis-', 'psisbar', 'psis+']
    phiList = ['phis-', 'phisbar', 'phis+']
    utilList = ['U_dc', 'U_du', 'U_ac', 'U_au'] 
    utilArray = np.vstack((U_dc, U_du, U_ac, U_au)).astype('|S6')
    util = {}
    for col in range(N):
        util[col] = '\n'.join([it[0]+'='+str(it[1].decode('utf-8')) \
            for it in list(zip(utilList,utilArray[:,col]))])
    nx.set_node_attributes(G, util, 'util')
    xVar = np.array(variables[:6*N]).reshape((-1,N))
    rearrangeX = np.vstack((xVar[4], xVar[3], xVar[5]))
    psiVar = np.array(variables[6*N:9*N]).reshape((-1,N))
    psiVar = (rearrangeX - psiVar).astype('|S4')
    phiVar = np.array(variables[9*N:12*N]).reshape((-1,N))
    phiVar = (rearrangeX - phiVar).astype('|S4')
    xVar = xVar.astype('|S4')
    x = {}
    psi = {}
    phi = {}
    for col in range(N):
        x[col] = '\n'.join([it[0]+'='+str(it[1].decode('utf-8')) \
            for it in list(zip(xList,xVar[:,col]))])
        psi[col] = '\n'.join([it[0]+'='+str(it[1].decode('utf-8')) \
            for it in list(zip(psiList,psiVar[:,col]))])
        phi[col] = '\n'.join([it[0]+'='+str(it[1].decode('utf-8')) \
            for it in list(zip(phiList,phiVar[:,col]))])

    nx.set_node_attributes(G, x, 'x')
    nx.set_node_attributes(G, psi, 'psi')
    nx.set_node_attributes(G, phi, 'phi')

    return G

def display_annotated_graph(G, G1, G2):
    """
    title::
        display_annotated_graph

    description::
        This method will display 8 graphs annotated with utilities and 
        variables, both with and without uncertainty.

    attributes::
        G
            Graph object (networkx) (original w/ nodes, utilities)
        G1
            Graph object (networkx) (with uncertainty)
        G2
            Graph object (networkx) (ignored uncertainty)

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
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Nodes')
    nx.draw_kamada_kawai(G, font_size=10, with_labels=True)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Utilities')
    nx.draw_kamada_kawai(G, 
                         labels=nx.get_node_attributes(G, 'util'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('X, with uncertainty')
    nx.draw_kamada_kawai(G1, 
                         labels=nx.get_node_attributes(G1, 'x'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('X, without uncertainty')
    nx.draw_kamada_kawai(G2, 
                         labels=nx.get_node_attributes(G2, 'x'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Psi, with uncertainty')
    nx.draw_kamada_kawai(G1, 
                         labels=nx.get_node_attributes(G1, 'psi'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Psi, without uncertainty')
    nx.draw_kamada_kawai(G2, 
                         labels=nx.get_node_attributes(G2, 'psi'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Phi, with uncertainty')
    nx.draw_kamada_kawai(G1, 
                         labels=nx.get_node_attributes(G1, 'phi'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Phi, without uncertainty')
    nx.draw_kamada_kawai(G2, 
                         labels=nx.get_node_attributes(G2, 'phi'), 
                         font_size=10)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()

    matplotlib.pyplot.show()

def generate_defender_utilities_plot(resultsGamma, 
                                     resultsUncert, 
                                     U_dc,
                                     U_du,
                                     U_ac,
                                     U_au,
                                     save=False, 
                                     show=False):
    """
    title::
        generate_defender_utilities_plot

    description::
        This method will plot 3c and 3d, and provide other stats used in the
        paper (ttests and percent change).

        NOTE: Only plots/reports results if there are values other than zero 
        in the array. If all values are zero, it does not plot/report results.
        Furthermore, if there are any None values from solutions, they will be
        reported but then ignored for producing results.

    attributes::
        resultsGamma
            Abbreviations: objective value = OV, recalculated (i.e., computed
            as though no uncertainty, recalculated with true uncertainty) = 
            recalc, original (i.e., computed as though no uncertainty) = orig,
            target (i.e., attacker best response target) = tgt, eta (i.e., 
            attacker best response behavior) = eta

            Depth: both OV, recalc OV, orig OV, both tgt, recalc tgt, orig 
            tgt, both eta, both recalc, both orig, CONDITIONAL PROBABILITY MEAN

            Rows: Results for graphs

            Columns: Results for gammas
        resultsUncert
            Same as above, but no conditional probability mean (therefore, only 9 
            columns), and kappa instead of gamma
        U_dc
            U_+^d (defender utility when defender successfully protects target)
        U_du
            U_-^d (defender utility when defender fails to protect target)
        U_ac
            U_+^a (attacker utility when defender successfully protects target)
        U_au
            U_-^a (attacker utility when defender fails to protect target)
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is False] NOTE: do not set both save and show to True, one may be 
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
    #NOTE: Warnings may arise if ttest or mean on array of all zeros (which may
    #happen with -nu or -ng options, for example). Check for these first.
    checkZeroUncert = True
    checkZeroGamma = True
    if np.all(resultsUncert == 0):
        checkZeroUncert = False
    if np.all(resultsGamma == 0):
        checkZeroGamma = False

    #NOTE: Ignoring nan's and computing without them.
    countNanUncert = np.where(np.isnan(resultsUncert) == True)
    countNanGamma = np.where(np.isnan(resultsGamma) == True)
    if countNanUncert[0].shape[0] > 0:
        print('nan exists in resultsUncert: ', countNanUncert)
    if countNanGamma[0].shape[0] > 0:
        print('nan exists in resultsGamma: ', countNanGamma)
    
    #t-test comparing optimal objective value and that from ignoring.
    if checkZeroUncert:
        ttest3d = stats.ttest_ind(resultsUncert[:,:,0], 
                                resultsUncert[:,:,1], 
                                axis=0,
                                nan_policy='omit')
        print('ttest 3d: ', ttest3d)
    if checkZeroGamma:
        ttest3c = stats.ttest_ind(resultsGamma[:,:,0], 
                                resultsGamma[:,:,1], 
                                axis=0,
                                nan_policy='omit')
        print('ttest 3c: ', ttest3c)
    

    #Mean over different graphs (print where nan, but ignore for mean).
    if checkZeroUncert:
        resultsUncertMean = np.nanmean(resultsUncert, axis=0)
    if checkZeroGamma:
        resultsGammaMean = np.nanmean(resultsGamma, axis=0)

    #Get maximum (absolute value) utility.
    maxUtil = max(np.abs(U_dc).max(), 
                  np.abs(U_du).max(), 
                  np.abs(U_ac).max(), 
                  np.abs(U_au).max())
    
    #Calculate percent change in objective including, ignoring uncertainty.
    if checkZeroUncert:
        percentChangeObsWith = calculate_percent_change(resultsUncertMean[:,0]\
            / maxUtil)
        percentChangeObsWithout = calculate_percent_change( \
            resultsUncertMean[:,1] / maxUtil)
        print('percent change 3d including, ignoring uncertainty: ', 
          percentChangeObsWith, percentChangeObsWithout)
    if checkZeroGamma:
        percentChangeDetWith = calculate_percent_change(resultsGammaMean[:,0] \
            / maxUtil)
        percentChangeDetWithout = calculate_percent_change( \
            resultsGammaMean[:,1] / maxUtil)
        print('percent change 3c including, ignoring uncertainty: ', 
            percentChangeDetWith, percentChangeDetWithout)
    

    #Plot varying uncertainty matrix. (3d)
    if checkZeroUncert:
        matplotlib.pyplot.rcParams.update({'font.size': 22})
        matplotlib.pyplot.figure()
        matplotlib.pyplot.xlabel('Amount of Observational Uncertainty'+\
                                 ' ($\kappa$)')
        matplotlib.pyplot.ylabel('Defender Expected Utility')
        matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsUncertMean[:,0], 
                            label='GUARDSS')
        matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsUncertMean[:,1], 
                            '^--',label='Def. Ignoring\nObservational '+\
                                        'Uncertainty')
        matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1.01, .102), loc=3,
            ncol=1, mode="expand", borderaxespad=0.)
        if save:
            matplotlib.pyplot.savefig('3d.pdf',
                                    bbox_inches='tight')

    #Plot varying gamma. (3c)
    if checkZeroGamma:
        matplotlib.pyplot.figure()
        matplotlib.pyplot.xlabel('Amount of Detection Uncertainty ($\gamma$)')
        matplotlib.pyplot.ylabel('Defender Expected Utility')
        matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsGammaMean[:,0], 
                            label='GUARDSS')
        matplotlib.pyplot.plot(np.arange(0,1,0.1), resultsGammaMean[:,1], 
                            '^--',label='Def. Ignoring Detection\nUncertainty')
        matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=1, mode="expand", borderaxespad=0.)
        if save:
            matplotlib.pyplot.savefig('3c.pdf', 
                                    bbox_inches='tight')
    if show:
        matplotlib.pyplot.show()
    
def generate_conditional_prob_plot(gammas, 
                                   resultsGamma, 
                                   save=True, 
                                   show=False):
    """
    title::
        generate_conditional_prob_plot

    description::
        This method will plot 3e, and provide other stats used in the paper
        (ttests).

    attributes::
        gammas
            The gammas that should be plotted (e.g., np.arange(0,1,0.1)).
        resultsGamma
            Abbreviations: objective value = OV, recalculated (i.e., computed
            as though no uncertainty, recalculated with true uncertainty) = 
            recalc, original (i.e., computed as though no uncertainty) = orig,
            target (i.e., attacker best response target) = tgt, eta (i.e., 
            attacker best response behavior) = eta

            Depth: both OV, recalc OV, orig OV, both tgt, recalc tgt, orig 
            tgt, both eta, both recalc, both orig, CONDITIONAL PROBABILITY MEAN

            Rows: Results for graphs

            Columns: Results for gammas
        save
            Set to True if you wish to save plots as files [default is False]
        show
            Set to True if you wish to show the plots (i.e., plt.show) [default
            is False] NOTE: do not set both save and show to True, one may be 
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
    #Plot conditional probability. (3e)
    matplotlib.pyplot.rcParams.update({'font.size':22})
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xlabel('Amount of Detection Uncertainty ($\gamma$)')
    matplotlib.pyplot.ylabel('P(fake | strong signal)')
    matplotlib.pyplot.plot(gammas, np.mean(resultsGamma[:,:,9], axis=0))
    matplotlib.pyplot.tight_layout()
    if save:
        matplotlib.pyplot.savefig('3e.pdf')
    if show:
        matplotlib.pyplot.show()

    print('ttest 3e',stats.ttest_ind(resultsGamma[:,0,9], resultsGamma[:,9,9], 
                                     axis=0))

def calculate_percent_change(defEU):
    """
    title::
        calculate_percent_change

    description::
        This method will calculate the percent change used to describe 3c and 
        3d in the paper. The input should be normalized by the maximum utility
        to get the reported results (see generate_defender_utilities_plot).

    attributes::
        defEU
            Array of objective values at different uncertainty levels. i.e., 1
            row of objective values, columns with increasing uncertainty (see
            generate_defender_utilities_plot, where we use the mean objective
            values over NUMBEROFGRAPH, and either the solution with uncertainty
            or the recalculated solution when uncertainty was ignored, as
            plotted in 3c/3d)

    returns::
        Percent change

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
    first = defEU[0]
    last = defEU[-1]
    return ((first - last) / first) * 100

def calculate_percent_difference(defEU):
    """
    title::
        calculate_percent_difference

    description::
        This method will calculate the percent difference.

    attributes::
        defEU
            Array of objective values at different uncertainty levels. i.e., 1
            row of objective values, columns with increasing uncertainty

    returns::
        Percent difference

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
    first = defEU[0]
    last = defEU[-1]
    return (np.abs(first - last) / ((first + last) / 2.)) * 100

def calculate_conditional_prob(x, phi, psi, gamma, tgt):
    """
    title::
        calculate_conditional_prob

    description::
        This method will calculate the conditional probability used in 3e in 
        the paper. The full equation is provided in Appendix C. Note that 
        "variables" in attributes are final optimal decision variable values
        returned by nonzeroColumnMethodRelaxation.

    attributes::
        x
            x variables (np.array(variables[:6*N]).reshape((-1,N)))
        phi
            phi variables (np.array(variables[9*N:12*N]).reshape((-1,N)))
        psi
            psi variables (np.array(variables[6*N:9*N]).reshape((-1,N)))
        gamma
            False negative rate
        tgt
            Current target for which to calculate

    returns::
        Conditional probability (see Appendix C) or 0 if undefined

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
    rearrangeX = np.vstack((x[4],x[3],x[5]))
    psiVar = rearrangeX - psi
    phiVar = rearrangeX - phi
    jointProbGamma = gamma * phiVar[:,tgt].sum()
    jointProb1mGamma = (1-gamma) * psiVar[:,tgt].sum()
    if (jointProbGamma + jointProb1mGamma >= -1e-5) and \
       (jointProbGamma + jointProb1mGamma <= 1e-5):
        return 0
    else:
        return jointProbGamma / (jointProbGamma + jointProb1mGamma)

def calculate_statistics(no, uncert, gamma=0.5):
    """
    title::
        calculate_statistics

    description::
        This method will calculate the statistics that appear in Section 6 in 
        the paper, namely, the mean probability and statistical significance of
        a sensor at the best target, and the mean probability and statistical 
        signficiance of state s- at the best target. Gamma = 0.5 was used in
        the paper, and is therefore the default. We also calculated other 
        potential values of interest, but print only those used in the paper.

    attributes::
        no
            Variables at best target e.g., variables[(5*N)+rtgt], for each 
            state, ignoring uncertainty, as follows:
                graph 1, graph 2, ...
            s+
            n+
            p
            n-
            sbar
            s-
            np.vstack((splusNo, nplusNo, ppNo, nminusNo, sbarNo, sminusNo))
        uncert
            Same as above but with uncertainty included
            np.vstack((splus, nplus, pp, nminus, sbar, sminus))
        gamma
            False negative rate [default is 0.5, as in paper]

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
    probSensorNo = no[0] + no[4] + no[5]
    probSensor = uncert[0] + uncert[4] + uncert[5]
    probCoveredNo = gamma * no[5] + no[0] + no[2] + no[1]
    probCovered = gamma * uncert[5] + uncert[0] + uncert[2] + uncert[1]
    probHumanNo = no[0] + no[2] + no[1]
    probHuman = uncert[0] + uncert[2] + uncert[1]
    probSensorPNo = no[0]
    probSensorP = uncert[0]
    probSensorMNo = no[5]
    probSensorM = uncert[5]

    probSensorNoMean = np.mean(probSensorNo)
    probSensorMean = np.mean(probSensor)
    ttestProbSensor = stats.ttest_ind(probSensorNo, probSensor)

    probCoveredNoMean = np.mean(probCoveredNo)
    probCoveredMean = np.mean(probCovered)
    ttestProbCovered = stats.ttest_ind(probCoveredNo, probCovered)

    probHumanNoMean = np.mean(probHumanNo)
    probHumanMean = np.mean(probHuman)
    ttestHuman = stats.ttest_ind(probHumanNo, probHuman)

    probSPlusNoMean = np.mean(probSensorPNo)
    probSPlusMean = np.mean(probSensorP)
    ttestSPlus = stats.ttest_ind(probSensorPNo, probSensorP)

    probSMinusNoMean = np.mean(probSensorMNo)
    probSMinusMean = np.mean(probSensorM)
    ttestSMinus = stats.ttest_ind(probSensorMNo, probSensorM)

    print('mean probability of sensor at best target including, ignoring uncertainty: ', probSensorMean, probSensorNoMean)
    print('ttest: ', ttestProbSensor)
    print('mean probability of state s- at best target including, ignoring uncertainty: ', probSMinusMean, probSMinusNoMean)
    print('ttest: ', ttestSMinus)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run defender utility '+\
                                                 'experiments for 3c, 3d, 3e.')
    parser.add_argument('N', type=int,
                        help='Number of targets (graph size).')
    parser.add_argument('k', type=int,
                        help='Number of human patrollers.')
    parser.add_argument('l', type=int,
                        help='Number of drones.')
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
    parser.add_argument('-a', '--annotateGraph',
                        help='Include if you wish to view annotated graphs.',
                        default=False,
                        action='store_true')
    parser.add_argument('-nv', '--saveVariables',
                        help='"No saving": Include if you DO NOT wish to '+\
                             'calculate Section 6 statistics.',
                        default=True,
                        action='store_false')
    parser.add_argument('-rt', '--robustnessTest',
                        help='Include to test for slightly incorrect gammas.',
                        default=False,
                        action='store_true')
    parser.add_argument('-ng', '--testGamma',
                        help='"No gamma": Include if you DO NOT wish to '+\
                             'test detection uncertainty.',
                        default=True,
                        action='store_false')
    parser.add_argument('-nu', '--testUMat',
                        help='"No uMat": Include if you DO NOT wish to '+\
                             'test observational uncertainty.',
                        default=True,
                        action='store_false')
    parser.add_argument('-sp', '--savePickle',
                        help='Include to save graphs, utilities, strategies.',
                        default=False,
                        action='store_true')
    parser.add_argument('-lp', '--loadPickle',
                        help='Include to load graphs, utilities, strategies.',
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
    parser.add_argument('-nr', '--runWithOneUncertainty',
                        help='"No run w/ 1": Include if you wish to run '+\
                             'with more than one uncertainty at a time.',
                        default=True,
                        action='store_false')
    parser.add_argument('-cg', '--graphType',
                        help='Include to use cycle graphs.',
                        default='watts',
                        action='store_const',
                        const='cycle')
    parser.add_argument('-g', '--NUMBEROFGRAPH', 
                        help='Number of graphs to test.',
                        type=int,
                        default=20)
    parser.add_argument('-o', '--runOneGammaAndUMat',
                        help='Include to run for only 1 uncertainty level.',
                        default=False,
                        action='store_true')
    parser.add_argument('-sr', '--saveResults',
                        help='Include to save the final result arrays as '+\
                             'pickles.',
                        default=False,
                        action='store_true')
    
    args = parser.parse_args()

    resultsGamma, resultsUncert = \
        compare_defender_utilities_script(args.N,
                                          args.k,
                                          args.l,
                                          plot=args.plot,
                                          save=args.save,
                                          show=args.show,
                                          annotateGraph=args.annotateGraph,
                                          saveVariables=args.saveVariables,
                                          robustnessTest=args.robustnessTest,
                                          testGamma=args.testGamma,
                                          testUMat=args.testUMat,
                                          savePickle=args.savePickle,
                                          loadPickle=args.loadPickle,
                                          loadGraph=args.loadGraph, 
                                          saveGraph=args.saveGraph,
                                          randomSeed=args.randomSeed,
                                          warmUp=args.warmUp,
                                          runWithOneUncertainty=args.runWithOneUncertainty,
                                          graphType=args.graphType,
                                          NUMBEROFGRAPH=args.NUMBEROFGRAPH, 
                                          runOneGammaAndUMat=args.runOneGammaAndUMat,
                                          saveResults=args.saveResults)