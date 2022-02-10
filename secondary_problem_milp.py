import networkx as nx
import cplex

def secondaryProblemMILPSolver(aas, bs, cs, ds, es, fs, G, N, k, l):
    """
    title::
        secondaryProblemMILPSolver

    description::
        Solves the secondary problem MILP to determine the next pure strategy to 
        add for column generation. See relax.py for use. More details can be 
        found in Section 5.2. Note that this has been renamed from the historical
        terminology.

    attributes::
        aas
            Dual variable(s) at locations with patrollers (p)
        bs
            Dual variable(s) at locations with nothing, patroller matched (n+)
        cs
            Dual variable(s) at locations with nothing, no patroller matched 
            (n-)
        ds
            Dual variable(s) at locations with drone, no patrollers nearby 
            (sbar)
        es
            Dual variable(s) at locations with drone, no patroller matched (s-)
        fs
            Dual variable(s) at locations with drone with patroller matched 
            (s+)
        G
            Graph object (networkx)
        N
            Number of targets (graph size)
        k
            Number of human patrollers
        l
            Number of drones
    
    returns::
        aSet
            Patroller locations for pure strategy to add (in a set)
        bSet
            Nothing (with patroller matched) locations for pure strategy to add
            (in a set)
        cSet
            Nothing (with no patroller matched) locations for pure strategy to 
            add (in a set)
        dSet
            Drone (with no patrollers nearby) locations for pure strategy to 
            add (in a set)
        eSet
            Drone (with no patroller matched) locations for pure strategy to 
            add (in a set)
        fSet
            Drone (with patroller matched) locations for pure strategy to add 
            (in a set)
        objectiveValue
            Optimal objective value from secondary problem (helps determine 
            convergence)
    
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
    cpx2 = cplex.Cplex()
    cpx2.set_log_stream(None)
    cpx2.set_error_stream(None)
    cpx2.set_warning_stream(None)
    cpx2.set_results_stream(None)

    #Maximize.
    cpx2.objective.set_sense(cpx2.objective.sense.maximize)

    #Directed graph instead.
    Gdir = nx.to_directed(G)
    Gdir_edges = Gdir.edges()
    edges = list(Gdir_edges)
    numEdges = len(Gdir_edges)

    #Objective function. Equation 10 (and 19 & 20)
    names2 = ["asecondary{0}".format(i+1) for i in range(N)] + \
             ["bsecondary{0}".format(i+1) for i in range(N)] + \
             ["csecondary{0}".format(i+1) for i in range(N)] + \
             ["dsecondary{0}".format(i+1) for i in range(N)] + \
             ["esecondary{0}".format(i+1) for i in range(N)] + \
             ["fsecondary{0}".format(i+1) for i in range(N)] + \
             ["ye{0}".format(i+1) for i in range(numEdges)]
    cpx2.variables.add(obj=aas + bs + cs + ds + es + fs + [0]*numEdges,
                       lb=[0]*(6*N+numEdges),
                       ub=[1]*(6*N+numEdges),
                       types=[cpx2.variables.type.binary]*(6*N+numEdges),
                       names=names2)

    #Equation 11
    cpx2.linear_constraints.add(lin_expr=[[range(N), [1]*N]],
                                senses=['L'],
                                rhs=[k])

    #A * v1 - v2 - v5 - v6 >= 0
    #Equation 14
    for i in range(N):
        neighborNodes = G[i].keys()
        cpx2.linear_constraints.add(lin_expr=[[list(neighborNodes) + [N+i] + \
                                               [4*N+i] + [5*N+i],
                                               [1]*len(neighborNodes) + [-1] + \
                                               [-1] + [-1]]],
                                    senses=['G'],
                                    rhs=[0])

    #Equation 15
    for i in range(N):
        neighborNodes = G[i].keys()
        cpx2.linear_constraints.add(lin_expr=[[list(neighborNodes) + [i] + \
                                               [N+i] + \
                                               [2*N+i] + [4*N+i] + [5*N+i],
                                               [1]*len(neighborNodes) + [-1] + \
                                               [-1] + \
                                               [-1] + [-1] + [-1]]],
                                    senses=['L'],
                                    rhs=[0])        

    #For each target, there is only one state.
    #Equation 13
    for i in range(N):
        cpx2.linear_constraints.add(lin_expr=[[[i, i+N, i+2*N, i+3*N,
                                                i+4*N, i+5*N],
                                               [1, 1, 1, 1, 1, 1]]],
                                    senses=['E'],
                                    rhs=[1])

    #The number of mobile sensors is at most l. 
    #NOTE: We use = to use all drones because there is no cost to use
    #all of the drones. To provide some brief intuition as to why, we do not 
    #include an explicit cost for using drones within the model, but we can 
    #imagine one scenario where using a drone could hurt us is if one of the 
    #drones is broken (e.g., failing to detect or signal). However, we are 
    #optimizing for each location, so we can address this within the optimization
    #(e.g., never signal if fully broken).
    #Equation 12
    cpx2.linear_constraints.add(lin_expr=[[list(range(3*N,6*N)), [1]*(3*N)]],
                                senses=['E'],
                                rhs=[l])

    #If target is in state p, must have one outgoing y = 1. 
    #Only a 1 if there is a p.
    #Equation 16
    for i in range(N):
        #Get indices of the out edges in y for first argument.
        oI = list(Gdir.out_edges(i))
        indices = [edges.index(x) for x in oI]

        #Add each of these (y1 + y2 + ... - v^p = 0).
        cpx2.linear_constraints.add(lin_expr=[[[6*N + x for x in indices] +[i],
                                               [1]*len(oI) + [-1]]],
                                    senses=['E'],
                                    rhs=[0])

    #Equation 16.5
    #NOTE: Missing from original conference paper writeup, but always present
    #in code. It has been added as a footnote in the paper version on our 
    #website.
    for j in range(N):
        #Get indices of the in edges in y for first argument.
        iI = list(Gdir.in_edges(j))
        indices = [edges.index(x) for x in iI]

        #Add each of these (y1 + y2 + ... - v^n+ -v^s+ >= 0).
        cpx2.linear_constraints.add(lin_expr=[[[6*N + x for x in indices] +\
                                               [j+N] + [j+5*N],
                                               [1]*len(iI) + [-1] + [-1]]],\
                                    senses=['G'],
                                    rhs=[0])

    #If target is not in s+ or n+, y must = 0. 
    #Equation 17
    for j in range(N):
        for currentEdge in Gdir.in_edges(j):
            index = edges.index(currentEdge)
            cpx2.linear_constraints.add(lin_expr=[[[j+N]+[j+5*N]+[6*N+index],
                                                   [1]+[1]+[-1]]],\
                                        senses = ['G'],
                                        rhs = [0])
                                                  

    #Solve.
    cpx2.solve()
    objectiveValue = cpx2.solution.get_objective_value()
    cpx2_variables = cpx2.solution.pool.get_values(0)

    #Get new targets to add.
    aSet = [i for i in range(N) if cpx2_variables[i] > 0.99]
    bSet = [i-N for i in range(N, 2*N) if cpx2_variables[i] > 0.99]
    cSet = [i-2*N for i in range(2*N, 3*N) if cpx2_variables[i] > 0.99]
    dSet = [i-3*N for i in range(3*N, 4*N) if cpx2_variables[i] > 0.99]
    eSet = [i-4*N for i in range(4*N, 5*N) if cpx2_variables[i] > 0.99]
    fSet = [i-5*N for i in range(5*N, 6*N) if cpx2_variables[i] > 0.99]

    return aSet, bSet, cSet, dSet, eSet, fSet, objectiveValue
