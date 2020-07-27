import numpy as np
import time
from random_strategy import randomStrategy
from random_strategy import greedyInitialStrategy
from slave_problem_milp import slaveProblemMILPSolver
import copy
from both_detection_fn_signaling_uncertainty \
    import nonzeroCplexSolverFixedTarget

MUTE = 1

def nonzeroColumnMethodRelaxation(N, 
                                  k, 
                                  l, 
                                  G, 
                                  gamma,
                                  eta,
                                  uMat,
                                  U_dc, 
                                  U_du, 
                                  U_ac, 
                                  U_au, 
                                  initialStrategies,
                                  keepStrategies=False,
                                  warmUpPlus=False,
                                  randomSeed=False,
                                  maxIteration=500):
    """
    title::
        nonzeroColumnMethodRelaxation

    description::
        Main branch and price solution method, using relaxation. NOTE: If it
        exceeds 1 hour, will return best solution so far. You can return None
        in line 290 if you wish to handle it that way instead, or the timeout
        time in line 198 and 289.

    attributes::
        N
            Number of targets (graph size)
        k
            Number of human patrollers
        l
            Number of drones
        G
            Graph object (networkx)
        gamma
            False negative rate
        eta
            Vector that depicts attacker behavior for each observation 
            {n, \sigma_0, \sigma_1} \in \Omega, where 1 represents attacking, 
            and 0 represents running away. So, \eta = 1 means an attacker will
            attack no matter what signaling state is observed, and \eta = 0
            means an attacker will never attack
        uMat
            Uncertainty matrix \Pi will contain the conditional probability 
            Pr[\omega^|\omega] for all \omega^, \omega \in \Omega
            to describe how likely the attacker will observe a signaling
            state \omega^ given the true signaling state is \omega
        U_dc
            U_+^d (defender utility when defender successfully protects target)
        U_du
            U_-^d (defender utility when defender fails to protect target)
        U_ac
            U_+^a (attacker utility when defender successfully protects target)
        U_au
            U_-^a (attacker utility when defender fails to protect target)
        initialStrategies
            Initial pure strategies (see paper section 5)
        keepStrategies
            Set to True to use the same pure strategies throughout every 
            iteration, i.e., run column generation from initial strategies, 
            then continue to add in one iteration, then use the full strategies
            found in this round as initial strategies for the next, and 
            continue this process of adding pure strategies - tends to run 
            quite slowly [default False]
        warmUpPlus
            Set to True to add additional greedy strategies to strategies 
            (whenever stuck at an objective value of -infinity) [default False]
        randomSeed
            Set to True to use random seeds to add random strategies to strategies 
            (whenever stuck at an objective value of -infinity) [default False]
        maxIteration
            Maximum number of iterations for column generation [default 500]


    returns::
        obj
            Optimal objective value
        variables
            Optimal values of decision variables
        cpx
            CPLEX object with LP (could use this to get obj/variables 
            externally)
        count
            Final number of iterations of column generation
        strategies
            Final pure strategies
        bestEta  
            Optimal attacker behavior vector, \eta*
        bestTarget
            Optimal attacker target, t
    
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
    if randomSeed:
        randomSeedList = list(range(500+maxIteration))

    count = 0
    accuracy = 0.0001

    if MUTE != 1: print('calculate relaxed upper bound...')
    relaxationBoundList = []

    startTime = time.time()
    #Solve relaxations for each target/behavior.
    for ai in eta:
        for target in range(N):
            
            #no strategies = relaxed
            res, _, _ = nonzeroCplexSolverFixedTarget(None, 
                                                        G,
                                                        N,
                                                        k,
                                                        l,
                                                        gamma,
                                                        ai,
                                                        uMat,
                                                        U_dc,
                                                        U_du,
                                                        U_ac,
                                                        U_au,
                                                        target)
            relaxationBoundList.append(res)

    if MUTE != 1: print('relaxation upper bound:', relaxationBoundList)

    if MUTE != 1: print('whole column generation...')

    objectiveValueList = [-np.inf]*(N*len(eta))
    cpxList = [None]*(N*len(eta))

    currentMaxValue = -np.inf
    currentMaxIndex = -1

    #Get solving order from relaxation values.
    solvingOrder = np.argsort(relaxationBoundList)[::-1]

    #Initialize strategies.
    if keepStrategies:
        strategies = copy.deepcopy(initialStrategies)
    else:
        strategies = [[] for i in range(N*len(eta))]

    #Go through solving order, solve each necessary via column generation.
    trackEta = []
    trackTarget = []
    startTime = time.time()
    for index, INDEX in enumerate(solvingOrder):
        target = int(INDEX % N)
        ai = eta[INDEX // N]
        trackEta.append(ai)
        trackTarget.append(target)

        #Add initial strategies for this INDEX.
        if keepStrategies == False:
            for st in initialStrategies:
                strategies[INDEX].append(st)

        if MUTE != 1: print('solving target: {0}'.format(INDEX))
        if currentMaxValue > relaxationBoundList[INDEX]:
            if MUTE != 1: print("don't need to solve target "+\
                                "{0} because {1} > {2}"\
                  .format(INDEX, currentMaxValue, 
                          relaxationBoundList[INDEX]))
            continue

        count = 0
        randomGenerateCount = 0
        randomGenerateMaxCount = 300
        objectiveHistoryList = [-np.inf]
        while (count < maxIteration) and (time.time() - startTime < 3600):
            if keepStrategies:
                obj, variables, cpx = nonzeroCplexSolverFixedTarget(\
                                                    strategies, 
                                                    G,
                                                    N, 
                                                    k,
                                                    l,
                                                    gamma,
                                                    ai,
                                                    uMat,
                                                    U_dc, 
                                                    U_du, 
                                                    U_ac, 
                                                    U_au, 
                                                    target)
            else:
                obj, variables, cpx = nonzeroCplexSolverFixedTarget(\
                                                    strategies[INDEX], 
                                                    G,
                                                    N, 
                                                    k,
                                                    l,
                                                    gamma,
                                                    ai,
                                                    uMat,
                                                    U_dc, 
                                                    U_du, 
                                                    U_ac, 
                                                    U_au, 
                                                    target)

            #Add strategy randomly to "get out of rut"/start
            if obj == -np.inf:
                if warmUpPlus:
                    for st in greedyInitialStrategy(G,k,l,U_dc,U_du,U_ac,U_au):
                        strategies[INDEX].append(st)
                    
                if randomSeed:
                    addSt = randomStrategy(G, k, l, 
                                          randomSeed=randomSeedList[500+count])
                    strategies[INDEX].append(addSt)
                else:
                    addSt = randomStrategy(G,k,l)
                    strategies[INDEX].append(addSt)
                randomGenerateCount += 1
                if randomGenerateCount > randomGenerateMaxCount:
                    break
                count += 1
                continue
            
            #Else, use the dual variables/slave problem to find optimal
            objectiveHistoryList.append(obj)
            aas = cpx.solution.get_dual_values(['a{0}'.format(i) \
                                               for i in range(N)])
            bs = cpx.solution.get_dual_values(['b{0}'.format(i) \
                                               for i in range(N)])
            cs = cpx.solution.get_dual_values(['c{0}'.format(i) \
                                               for i in range(N)])
            ds = cpx.solution.get_dual_values(['d{0}'.format(i) \
                                               for i in range(N)])
            es = cpx.solution.get_dual_values(['e{0}'.format(i) \
                                               for i in range(N)])
            fs = cpx.solution.get_dual_values(['f{0}'.format(i) \
                                               for i in range(N)])
            r = cpx.solution.get_dual_values('r') #(sum pe's = 1 constraint)

            aSet, bSet, cSet, dSet, eSet, fSet, newObjectiveValue = \
                slaveProblemMILPSolver(aas, bs, cs, ds, es, fs, G, N, k, l)
            if keepStrategies:
                strategies.append((aSet, bSet, cSet, dSet, eSet, fSet))
            else:
                strategies[INDEX].append((aSet, bSet, cSet, dSet, eSet, \
                                            fSet))

            # =========== convergent criteria 2 ===============
            if (newObjectiveValue - r) < 0.01:
                if MUTE != 1: print('slave problem objective value: '+\
                                    '{0}, r: {1}'.format(\
                      newObjectiveValue, r))
                break

            count += 1


        if obj > currentMaxValue:
            currentMaxValue = obj
            currentMaxIndex = INDEX
            if MUTE != 1: print('found max value: {0} from index: {1}'\
                                .format(obj,INDEX))

        if time.time() - startTime > 3600:
            print('timeout', time.time() - startTime)

        objectiveValueList[INDEX] = obj
        cpxList[INDEX] = cpx

    #Return best.
    bestIndex = np.argmax(objectiveValueList)
    if MUTE != 1: print('best index:', bestIndex)
    obj = objectiveValueList[bestIndex]
    cpx = cpxList[bestIndex]
    if keepStrategies == False:
        strategies = strategies[bestIndex]
    bestEta = trackEta[bestIndex]
    bestTarget = trackTarget[bestIndex]

    if cpx:
        variables = cpx.solution.get_values()
        return obj, variables, cpx, count, strategies, bestEta, bestTarget
    else:
        return obj, None, None, None, None, None, None

