import networkx as nx
import numpy as np
import cplex
import sys


MUTE = 1

def nonzeroCplexSolverFixedTarget(strategies, 
                                  G,
                                  N,
                                  k,
                                  l, 
                                  gamma,
                                  eta,
                                  uMat,
                                  U_dc, 
                                  U_du, 
                                  U_ac, 
                                  U_au, 
                                  target):
    """
    title::
        nonzeroCplexSolverFixedTarget

    description::
        Solve the LP (partially described by (1-9) in the paper) for a single
        target. Please refer to Appendix for LP definition because LP in main
        body is detection only, while the Appendix has the full version with
        both detection and observational uncertainty. Note that U_-s^d is
        denoted by 1., U_sigma0^d is denoted by 2., U_sigma1^d is denoted by
        3., and U_omegahat^a is denoted by 4. in the paper. Also note that this
        code is used to run both the relaxed version and full version. In 
        CPLEX, senses=G means >=, L means <=, E means =. Please note that 
        sometimes you may get an infeasible solution to this LP (you can 
        print(cpx.solution.get_status_string()) to see when this happens), 
        particularly if a target is not possibly a best response for the 
        attacker.

    attributes::
        strategies
            Pure strategies (None if wanting to run relaxed version)
        G
            Graph object (networkx)
        N
            Number of targets (graph size)
        k
            Number of human patrollers
        l
            Number of drones
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
        target
            Current target for which to solve
    
    returns::
        solution
            final objective value
        variables
            final optimal decision variable values
        cpx
            CPLEX object
    
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
    uMat = uMat.copy()
    if strategies is not None: #NORMAL
        mode = 'normal'
        strategiesSize = len(strategies)
        variableSize = 15*N + strategiesSize

        names = ["xp{0}".format(i+1) for i in range(N)] + \
                ["xn+{0}".format(i+1) for i in range(N)] + \
                ["xn-{0}".format(i+1) for i in range(N)] + \
                ["xsbar{0}".format(i+1) for i in range(N)] + \
                ["xs-{0}".format(i+1) for i in range(N)] + \
                ["xs+{0}".format(i+1) for i in range(N)] + \
                ["psis-{0}".format(i+1) for i in range(N)] + \
                ["psisbar{0}".format(i+1) for i in range(N)] + \
                ["psis+{0}".format(i+1) for i in range(N)] + \
                ["phis-{0}".format(i+1) for i in range(N)] + \
                ["phisbar{0}".format(i+1) for i in range(N)] + \
                ["phis+{0}".format(i+1) for i in range(N)] + \
                ["bn{0}".format(i+1) for i in range(N)] + \
                ["bsigma1{0}".format(i+1) for i in range(N)] + \
                ["bsigma0{0}".format(i+1) for i in range(N)] + \
                ["qe{0}".format(i+1) for i in range(strategiesSize)]

    else: #RELAXATION (see Section 5.2, last paragraph)
        mode = 'relax'
        #Directed graph instead.
        Gdir = nx.to_directed(G)
        Gdir_edges = Gdir.edges()
        edges = list(Gdir_edges)
        numEdges = len(Gdir_edges)

        variableSize = 15*N + numEdges

        names = ["xp{0}".format(i+1) for i in range(N)] + \
                ["xn+{0}".format(i+1) for i in range(N)] + \
                ["xn-{0}".format(i+1) for i in range(N)] + \
                ["xsbar{0}".format(i+1) for i in range(N)] + \
                ["xs-{0}".format(i+1) for i in range(N)] + \
                ["xs+{0}".format(i+1) for i in range(N)] + \
                ["psis-{0}".format(i+1) for i in range(N)] + \
                ["psisbar{0}".format(i+1) for i in range(N)] + \
                ["psis+{0}".format(i+1) for i in range(N)] + \
                ["phis-{0}".format(i+1) for i in range(N)] + \
                ["phisbar{0}".format(i+1) for i in range(N)] + \
                ["phis+{0}".format(i+1) for i in range(N)] + \
                ["bn{0}".format(i+1) for i in range(N)] + \
                ["bsigma1{0}".format(i+1) for i in range(N)] + \
                ["bsigma0{0}".format(i+1) for i in range(N)] + \
                ["ye{0}".format(i+1) for i in range(numEdges)]

    while 1:
        cpx = cplex.Cplex()
  
        cpx.parameters.timelimit.set(3600.0)
        cpx.parameters.mip.limits.treememory.set(8192)
        cpx.parameters.mip.strategy.file.set(3)
        cpx.parameters.simplex.tolerances.feasibility.set(1e-9)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_results_stream(None)
  
        #Maximize objective function.
        cpx.objective.set_sense(cpx.objective.sense.maximize)

        #Objective function (equation 1 in Appendix).
        uQuiet = (uMat[0,1]*eta[0] + uMat[1,1]*eta[1] + uMat[2,1]*eta[2]) #p^a_sigma0
        uSignal = (uMat[0,2]*eta[0] + uMat[1,2]*eta[1] + uMat[2,2]*eta[2]) #p^a_sigma1
        if mode == 'normal':
            if MUTE != 1: print('uMat', uMat)
            if MUTE != 1: print('target', target)
            if MUTE != 1: print('eta', eta)
            if MUTE != 1: print('uQuiet', uQuiet)
            if MUTE != 1: print('uSignal', uSignal)

        objectiveValueCoef = np.zeros(variableSize)
        #U_-s^d
        objectiveValueCoef[0*N + target] = U_dc[target] #x_t^p * U_dc
        objectiveValueCoef[1*N + target] = U_dc[target] * eta[0] #x_t^n+ * U_dc
        objectiveValueCoef[2*N + target] = U_du[target] * eta[0] #x_t^n- * U_du
        
        #Combination of U_sigma1^d and U_sigma0^d.
        #x's in U_sigma1^d
        objectiveValueCoef[5*N + target] = U_dc[target] * uSignal
        objectiveValueCoef[4*N + target] =((1-gamma) * U_dc[target] * uSignal)+\
                                           (gamma * U_du[target] * uSignal)
        objectiveValueCoef[3*N + target] = U_du[target] * uSignal
        #U_dc psi_t^s+
        objectiveValueCoef[8*N + target] = ((1-gamma) * U_dc[target] * uQuiet)-\
                                           ((1-gamma) * U_dc[target] * uSignal)
        #U_dc psi_t^s-
        objectiveValueCoef[6*N + target] = ((1-gamma) * U_dc[target] * uQuiet)-\
                                           ((1-gamma) * U_dc[target] * uSignal) 
        #U_du psi_t^sbar
        objectiveValueCoef[7*N + target] = ((1-gamma) * U_du[target] * uQuiet)-\
                                           ((1-gamma) * U_du[target] * uSignal)
        #U_dc phi_t^s+
        objectiveValueCoef[11*N + target] = (gamma * U_dc[target] * uQuiet)-\
                                            (gamma * U_dc[target] * uSignal)
        #U_du phi_t^s-
        objectiveValueCoef[9*N + target] = (gamma * U_du[target] * uQuiet)-\
                                           (gamma * U_du[target] * uSignal)
        #U_du phi_t^sbar
        objectiveValueCoef[10*N + target] = (gamma * U_du[target] * uQuiet)-\
                                            (gamma * U_du[target] * uSignal)
        if mode == 'normal':
            if MUTE != 1: print('objectiveValueConf', list(objectiveValueCoef))

        #Bounds (Equations 5, 6, 8, 9 in Appendix LP and 19 & 20 in Section 5.2 for relaxed)
        #NOTE: Not binary like in slave_problem_milp because it's a relaxation.
        if mode == 'normal':
            lowerBounds = [0.0] * (15*N + strategiesSize)
            upperBounds = [1] * (12*N)
            upperBounds += [max(U_au)] *(3*N)
            upperBounds += [1] * (strategiesSize)
        else:
            lowerBounds = [0.0] * (15*N + numEdges)
            upperBounds = [1] * (12*N)
            upperBounds += [max(U_au)] *(3*N)
            upperBounds += [1] * (numEdges)

        #Add objective function.
        cpx.variables.add(obj = list(objectiveValueCoef), \
                          lb = lowerBounds, \
                          ub = upperBounds, \
                          names = names)
    
        
        if mode == 'normal':
            #Equation 2 from Appendix LP.
            pList = [[] for x in range(N)]
            npList = [[] for x in range(N)]
            nmList = [[] for x in range(N)]
            sbarList = [[] for x in range(N)]
            smList = [[] for x in range(N)]
            spList = [[] for x in range(N)]

            for i in range(strategiesSize):
                for aNode in set(strategies[i][0]):
                    pList[aNode].append(15*N+i)
                for bNode in set(strategies[i][1]):
                    npList[bNode].append(15*N+i)
                for cNode in set(strategies[i][2]):
                    nmList[cNode].append(15*N+i)  
                for dNode in set(strategies[i][3]):
                    sbarList[dNode].append(15*N+i)  
                for eNode in set(strategies[i][4]):
                    smList[eNode].append(15*N+i)  
                for fNode in set(strategies[i][5]):
                    spList[fNode].append(15*N+i)

            #(pure strategy * -1) + (x_i^theta) = 0
            for i in range(N):
                cpx.linear_constraints.add(lin_expr=[[pList[i] + [i    ], \
                                                [-1.0]*len(pList[i]) +[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['a{0}'.format(i)])
                cpx.linear_constraints.add(lin_expr=[[npList[i] + [N+i  ], \
                                                [-1.0]*len(npList[i])+[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['b{0}'.format(i)] )
                cpx.linear_constraints.add(lin_expr=[[nmList[i] + [2*N+i], \
                                                [-1.0]*len(nmList[i])+[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['c{0}'.format(i)])
                cpx.linear_constraints.add(lin_expr=[[sbarList[i] + [3*N+i], \
                                                [-1.0]*len(sbarList[i])+[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['d{0}'.format(i)])
                cpx.linear_constraints.add(lin_expr=[[smList[i] + [4*N+i], \
                                               [-1.0]*len(smList[i])+[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['e{0}'.format(i)])
                cpx.linear_constraints.add(lin_expr=[[spList[i] + [5*N+i], \
                                               [-1.0]*len(spList[i])+[1.0]]],\
                                           senses=['E'], \
                                           rhs=[0.0], \
                                           names=['f{0}'.format(i)])

            #Equation 3 from Appendix LP.
            #(sum qe's = 1)
            cpx.linear_constraints.add(\
                lin_expr=[[[15*N + i for i in range(strategiesSize)], \
                           [1.0]*strategiesSize]], \
                senses=['E'], \
                rhs=[1.0], \
                names=['r'])

            #Equation 4 from Appendix LP.
            #qe >= 0 constraint
            for i in range(15*N + strategiesSize):
                cpx.linear_constraints.add(lin_expr=[[[i], [1]]], \
                                           senses=['G'], \
                                           rhs=[0.0])

        else:
            #Equation 11 and 12 in Section 5.2 (but x instead of v).
            cpx.linear_constraints.add(lin_expr=[[range(N), [1]*(N)]],
                                       senses=['L'],
                                       rhs=[k])
            cpx.linear_constraints.add(lin_expr=[[range(3*N, 6*N), [1]*(3*N)]],
                                       senses=['L'],
                                       rhs=[l])
            #Equation 13 in Section 5.2 (but x instead of v).
            for i in range(N):
                cpx.linear_constraints.add(lin_expr=[[[i, N+i, 2*N+i, 3*N+i, \
                                                       4*N+i, 5*N+i],
                                                    [1.0,1.0,1.0,1.0,1.0,1.0]]],
                                           senses=['E'],
                                           rhs=[1.0])

            #A * x^p - x^n+ - x^s- - x^s+ >= 0
            #Equation 14 (but x instead of v) (added to original code)
            for i in range(N):
                neighborNodes = G[i].keys()
                cpx.linear_constraints.add(lin_expr=[[list(neighborNodes) + [N+i] + \
                                                    [4*N+i] + [5*N+i],
                                                    [1]*len(neighborNodes) + [-1] + \
                                                    [-1] + [-1]]],
                                            senses=['G'],
                                            rhs=[0])

            #Equation 15 (but x instead of v) (added to original code)
            for i in range(N):
                neighborNodes = G[i].keys()
                cpx.linear_constraints.add(lin_expr=[[list(neighborNodes) + [i] + \
                                                    [N+i] + \
                                                    [2*N+i] + [4*N+i] + [5*N+i],
                                                    [1]*len(neighborNodes) + [-1] + \
                                                    [-1] + \
                                                    [-1] + [-1] + [-1]]],
                                            senses=['L'],
                                            rhs=[0]) 

            #Equation 16 in Section 5.2 (but x instead of v).
            for i in range(N):
                #Get indices of the out edges in y for first argument.
                oI = list(Gdir.out_edges(i))
                index = [edges.index(x) for x in oI]

                #Add each of these (y1 + y2 + ... - x^p = 0).
                cpx.linear_constraints.add(
                    lin_expr = [[[15*N+x for x in index]+[i],
                                 [1]*len(oI) + [-1]]],
                    senses=['E'],
                    rhs=[0])

            #Equation 16.5 in slave_problem_milp.py (but x instead of v).
            for j in range(N):
                #Get indices of the in edges in y for first argument.
                iI = list(Gdir.in_edges(j))
                index = [edges.index(x) for x in iI]

                #Add each of these (y1 + y2 + ... - x^n+ -x^s+ >= 0).
                cpx.linear_constraints.add(lin_expr=[[[15*N+x for x in index]+\
                                                      [j+N] + [j+5*N],
                                                   [1]*len(iI) + [-1] + [-1]]],\
                                            senses=['G'],
                                            rhs=[0])

            #If target is not in s+ or n+, y must = 0. 
            #Equation 17 (but x instead of v) (added to original code)
            for j in range(N):
                for currentEdge in Gdir.in_edges(j):
                    index = edges.index(currentEdge)
                    cpx.linear_constraints.add(lin_expr=[[[j+N]+[j+5*N]+[15*N+index],
                                                        [1]+[1]+[-1]]],\
                                                senses = ['G'],
                                                rhs = [0])

            #Equation 4 from Appendix LP (but y instead of q). This isn't in 
            #paper, but is already covered by bounds so just for computation 
            #stability.
            for i in range(15*N + numEdges):
                cpx.linear_constraints.add(lin_expr=[[[i], [1]]], \
                                           senses=['G'], \
                                           rhs=[0])


        #Equation 7 in Appendix LP (note, the LHS is the same as the objective 
        #function, but for attackers instead of defenders).
        for i in range(N):
            if i == target:
                continue

            constraintCoef = [U_ac[target], #x_t^p * U_ac #positive targets
                              U_ac[target] * eta[0], #x_t^n+ * U_ac
                              U_au[target] * eta[0], #x_t^n- * U_au
                              U_ac[target] * uSignal, #x_t^s+
                              ((1-gamma) * U_ac[target] * uSignal)+\
                                  (gamma * U_au[target] * uSignal), #x_t^s-
                              U_au[target] * uSignal, #x_t^sbar
                              ((1-gamma) * U_ac[target] * uQuiet)-\
                                  ((1-gamma) * U_ac[target] * uSignal), #psi_t^s+
                              ((1-gamma) * U_ac[target] * uQuiet)-\
                                  ((1-gamma) * U_ac[target] * uSignal), #psi_t^s-
                              ((1-gamma) * U_au[target] * uQuiet)-\
                                  ((1-gamma) * U_au[target] * uSignal), #psi_t^sbar
                              (gamma * U_ac[target] * uQuiet)-\
                                  (gamma * U_ac[target] * uSignal), #phi_t^s+
                              (gamma * U_au[target] * uQuiet)-\
                                  (gamma * U_au[target] * uSignal), #phi_t^s-
                              (gamma * U_au[target] * uQuiet)-\
                                  (gamma * U_au[target] * uSignal), #phi_t^sbar
                              -U_ac[i], #x_i^p * U_ac
                              -1, #b_i^n
                              -1, #b_i^sigma0
                              -1] #b_i^sigma1

            indices = [0*N + target, #x_t^p
                       1*N + target, #x_t^n+
                       2*N + target, #x_t^n-
                       5*N + target, #x_t^s+
                       4*N + target, #x_t^s-
                       3*N + target, #x_t^sbar
                       8*N + target, #psi_t^s+
                       6*N + target, #psi_t^s-
                       7*N + target, #psi_t^sbar
                       11*N + target,#phi_t^s+
                       9*N + target, #phi_t^s-
                       10*N + target,#phi_t^sbar
                       0*N + i,      #x_i^p
                       12*N + i,     #b_i^n
                       13*N + i,     #b_i^sigma0
                       14*N + i]     #b_i^sigma1

            cpx.linear_constraints.add(\
                lin_expr=[[indices, constraintCoef]], 
                senses=['G'], 
                rhs=[0.0])

        #Equation 5 in Appendix LP (based on 4., which is similar to previous/
        #objective function after first 2 lines).
        #Ua <= b's
        for i in range(N):
            toAdd = [0,2,1]
            for state in range(3):
                constraintCoef = [U_ac[i]*uMat[state,0],
                                  U_au[i]*uMat[state,0],
                                  U_ac[i]*uMat[state,2],
                                  ((1-gamma) * U_ac[i]*uMat[state,2])+\
                                   (gamma*U_au[i]*uMat[state,2]),
                                  U_au[i]*uMat[state,2],
                                  ((1-gamma) * -U_ac[i] * uMat[state,2])+\
                                   ((1-gamma) * U_ac[i] * uMat[state,1]),
                                  ((1-gamma) * -U_ac[i] * uMat[state,2])+\
                                   ((1-gamma) * U_ac[i] * uMat[state,1]),
                                  ((1-gamma) * -U_au[i] * uMat[state,2])+\
                                   ((1-gamma) * U_au[i] * uMat[state,1]),
                                  (gamma * -U_ac[i] * uMat[state,2])+\
                                   (gamma * U_ac[i] * uMat[state,1]),
                                  (gamma * -U_au[i] * uMat[state,2])+\
                                   (gamma * U_au[i] * uMat[state,1]),
                                  (gamma * -U_au[i] * uMat[state,2])+\
                                   (gamma * U_au[i] * uMat[state,1]),
                                  -1]
                indices = [1*N + i, #x_i^n+
                           2*N + i, #x_i^n-
                           5*N + i, #x_i^s+
                           4*N + i, #x_i^s-
                           3*N + i, #x_i^sbar
                           8*N + i, #psi_i^s+
                           6*N + i, #psi_i^s-
                           7*N + i, #psi_i^sbar
                           11*N + i,#phi_i^s+
                           9*N + i, #phi_i^s-
                           10*N + i,#phi_i^sbar
                           (12+toAdd[state])*N + i] #b_i^omegahat

                cpx.linear_constraints.add(\
                lin_expr=[[indices, constraintCoef]],
                senses=['L'],
                rhs=[0.0])
 
        #Equation 10. Write out all 3 omega hats. Because of 
        #2*eta_t^omegahat - 1, changes the L or G based on value of 
        #eta_t^omegahat.
        #First, omega hat = n.
        constraintCoef = [U_ac[target] * uMat[0,0], #x_t^n+ * U_ac
                          U_au[target] * uMat[0,0], #x_t^n- * U_au
                          U_ac[target] * uMat[0,2], #x_t^s+
                          ((1-gamma) * U_ac[target] * uMat[0,2])+\
                            (gamma * U_au[target] * uMat[0,2]), #x_t^s-
                          U_au[target] * uMat[0,2], #x_t^sbar
                          ((1-gamma) * -U_ac[target] * uMat[0,2])+\
                            ((1-gamma) * U_ac[target] * uMat[0,1]), #psi_t^s+
                          ((1-gamma) * -U_ac[target] * uMat[0,2])+\
                            ((1-gamma) * U_ac[target] * uMat[0,1]), #psi_t^s-
                          ((1-gamma) * -U_au[target] * uMat[0,2])+\
                            ((1-gamma) * U_au[target] * uMat[0,1]), #psi_t^sbar
                          (gamma * -U_ac[target] * uMat[0,2])+\
                            (gamma * U_ac[target] * uMat[0,1]), #phi_t^s+
                          (gamma * -U_au[target] * uMat[0,2])+\
                            (gamma * U_au[target] * uMat[0,1]), #phi_t^s-
                          (gamma * -U_au[target] * uMat[0,2])+\
                            (gamma * U_au[target] * uMat[0,1])] #phi_t^sbar

        indices = [1*N + target, #x_t^n+
                   2*N + target, #x_t^n-
                   5*N + target, #x_t^s+
                   4*N + target, #x_t^s-
                   3*N + target, #x_t^sbar
                   8*N + target, #psi_t^s+
                   6*N + target, #psi_t^s-
                   7*N + target, #psi_t^sbar
                   11*N + target,#phi_t^s+
                   9*N + target, #phi_t^s-
                   10*N + target]#phi_t^sbar

        if eta[0] == 1:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['G'],
                    rhs=[0.0])
        else:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['L'],
                    rhs=[0.0])

        #Second, omega hat = sigma0.
        constraintCoef = [U_ac[target] * uMat[1,0], #x_t^n+ * U_ac
                          U_au[target] * uMat[1,0], #x_t^n- * U_au
                          U_ac[target] * uMat[1,2], #x_t^s+
                          ((1-gamma) * U_ac[target] * uMat[1,2])+\
                            (gamma * U_au[target] * uMat[1,2]), #x_t^s-
                          U_au[target] * uMat[1,2], #x_t^sbar
                          ((1-gamma) * -U_ac[target] * uMat[1,2])+\
                            ((1-gamma) * U_ac[target] * uMat[1,1]), #psi_t^s+
                          ((1-gamma) * -U_ac[target] * uMat[1,2])+\
                            ((1-gamma) * U_ac[target] * uMat[1,1]), #psi_t^s-
                          ((1-gamma) * -U_au[target] * uMat[1,2])+\
                            ((1-gamma) * U_au[target] * uMat[1,1]), #psi_t^sbar
                          (gamma * -U_ac[target] * uMat[1,2])+\
                            (gamma * U_ac[target] * uMat[1,1]), #phi_t^s+
                          (gamma * -U_au[target] * uMat[1,2])+\
                            (gamma * U_au[target] * uMat[1,1]), #phi_t^s-
                          (gamma * -U_au[target] * uMat[1,2])+\
                            (gamma * U_au[target] * uMat[1,1])] #phi_t^sbar

        indices = [1*N + target, #x_t^n+
                   2*N + target, #x_t^n-
                   5*N + target, #x_t^s+
                   4*N + target, #x_t^s-
                   3*N + target, #x_t^sbar
                   8*N + target, #psi_t^s+
                   6*N + target, #psi_t^s-
                   7*N + target, #psi_t^sbar
                   11*N + target,#phi_t^s+
                   9*N + target, #phi_t^s-
                   10*N + target]#phi_t^sbar

        if eta[1] == 1:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['G'],
                    rhs=[0.0])
        else:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['L'],
                    rhs=[0.0])

        #Third, omega hat = sigma1.
        constraintCoef = [U_ac[target] * uMat[2,0], #x_t^n+ * U_ac
                          U_au[target] * uMat[2,0], #x_t^n- * U_au
                          U_ac[target] * uMat[2,2], #x_t^s+
                          ((1-gamma) * U_ac[target] * uMat[2,2])+\
                            (gamma * U_au[target] * uMat[2,2]), #x_t^s-
                          U_au[target] * uMat[2,2], #x_t^sbar
                          ((1-gamma) * -U_ac[target] * uMat[2,2])+\
                            ((1-gamma) * U_ac[target] * uMat[2,1]), #psi_t^s+
                          ((1-gamma) * -U_ac[target] * uMat[2,2])+\
                            ((1-gamma) * U_ac[target] * uMat[2,1]), #psi_t^s-
                          ((1-gamma) * -U_au[target] * uMat[2,2])+\
                            ((1-gamma) * U_au[target] * uMat[2,1]), #psi_t^sbar
                          (gamma * -U_ac[target] * uMat[2,2])+\
                            (gamma * U_ac[target] * uMat[2,1]), #phi_t^s+
                          (gamma * -U_au[target] * uMat[2,2])+\
                            (gamma * U_au[target] * uMat[2,1]), #phi_t^s-
                          (gamma * -U_au[target] * uMat[2,2])+\
                            (gamma * U_au[target] * uMat[2,1])] #phi_t^sbar

        indices = [1*N + target, #x_t^n+
                   2*N + target, #x_t^n-
                   5*N + target, #x_t^s+
                   4*N + target, #x_t^s-
                   3*N + target, #x_t^sbar
                   8*N + target, #psi_t^s+
                   6*N + target, #psi_t^s-
                   7*N + target, #psi_t^sbar
                   11*N + target,#phi_t^s+
                   9*N + target, #phi_t^s-
                   10*N + target]#phi_t^sbar

        if eta[2] == 1:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['G'],
                    rhs=[0.0])
        else:
            cpx.linear_constraints.add(\
                    lin_expr=[[indices, constraintCoef]],
                    senses=['L'],
                    rhs=[0.0])
                
              

        
        #Equations 8 and 9 in Appendix LP.
        for i in range(N):
            cpx.linear_constraints.add(lin_expr=[[[3*N+i, 7*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])
            cpx.linear_constraints.add(lin_expr=[[[3*N+i, 10*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])
            cpx.linear_constraints.add(lin_expr=[[[5*N+i, 8*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])
            cpx.linear_constraints.add(lin_expr=[[[5*N+i, 11*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])
            cpx.linear_constraints.add(lin_expr=[[[4*N+i, 6*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])
            cpx.linear_constraints.add(lin_expr=[[[4*N+i, 9*N+i], \
                                                  [1.0, -1.0]]], \
                                       senses=['G'], \
                                       rhs=[0.0])

        
        #Write problem. Uncomment if you'd like to write LP file.
        #cpx.write('problem.lp')

        

        try:
            #Solve LP.
            cpx.solve()
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            solution = -np.inf
            variables = None
            cpx = None

        try:
            solution = cpx.solution.get_objective_value()
            variables = cpx.solution.get_values()
            if MUTE != 1: print(cpx.solution.get_status_string())
            
            #Check for errors with solution (e.g., not in bounds).
            testList = []
            if cpx.solution.get_status_string() != 'optimal':
                testList.append(None)

            if mode == 'normal':
                for i in range(15*N + strategiesSize):
                    if i in range(12*N,15*N):
                        continue
                    if (variables[i] < -0.00001) or (variables[i] > 1.00001):
                        testList.append(None)
            else:
                for i in range(15*N + numEdges):
                    if i in range(12*N,15*N):
                        continue
                    if variables[i] < -0.00001:
                        testList.append(None)
            if len(testList) > 0:
                solution = -np.inf
                variables = None
                cpx = None
        except:
            #Might just be no solution to this particular LP b/c not attacker 
            #best response
            solution = -np.inf
            variables = None
            cpx = None

        #Check for another error with solution.
        if variables is not None:
            if sum(variables[:12*N]) <= 0:
                solution = -np.inf
                variables = None
                cpx = None

        del uMat

        return solution, variables, cpx