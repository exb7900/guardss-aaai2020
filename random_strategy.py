import itertools
import random
import networkx as nx

MUTE = 1

def randomStrategy(G, k, l, randomSeed=None):
    """
    title::
        randomStrategy

    description::
        Generate random initial strategy. See compare_heuristic_script.py 
        for example use.

    attributes::
        G
            Graph object (networkx)
        k
            Number of human patrollers
        l
            Number of drones
        randomSeed
            If integer is provided, use as random seed for randomly sampling
            nodes; if None, no random seed is used [default is None]
    
    returns::
        Initial strategy (sets of targets in a list)
    
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
    nodes = set(G.nodes())
    if randomSeed is not None:
        random.seed(randomSeed)
    aSet = set(random.sample(nodes, k)) #patrollers are here (a)
    if MUTE != 1: print('aSet', aSet)
    adjacentList = [] #all neighboring nodes to patrollers
    chosenList = [] #choose one of the adjacent nodes to visit upon neg. det.
    for xx in aSet:
        outEdges = list(G[xx].keys())
        if MUTE != 1: print(outEdges)
        adjacentList += outEdges #all neighbors
        if randomSeed is not None:
            random.seed(randomSeed)
        add = random.sample(outEdges, 1)
        if MUTE != 1: print('outedges', add)
        chosenList += add #choose 1 for ranger    

    adjacentSet = set(adjacentList)
    remainingNodes = nodes - aSet #nodes with no patrollers
    if randomSeed is not None:
        random.seed(randomSeed)
    droneSet = set(random.sample(remainingNodes, l)) #place drones
    if MUTE != 1: print('droneSet', droneSet)
    dSet = droneSet - adjacentSet #drone, no patrollers nearby (d)
    emptyNodes = remainingNodes - droneSet #nothing at these nodes
    bSet = emptyNodes.intersection(chosenList) #nothing, patroller matched (b)
    cSet = emptyNodes - bSet #nothing, no patroller matched (c)
    fSet = droneSet.intersection(chosenList) #drone with patroller matched (f)
    eSet = droneSet - fSet - dSet #drone, no patroller matched (e)
    return [aSet,
            bSet,
            cSet,
            dSet,
            eSet,
            fSet]


def greedyInitialStrategy(G, k, l, U_dc,U_du,U_ac,U_au):
    """
    title::
        greedyInitialStrategy

    description::
        Generate greedy initial strategies (in paper, this is mentioned as the 
        "warm-up" technique). See compare_heuristic_script.py for example use.

    attributes::
        G
            Graph object (networkx)
        k
            Number of human patrollers
        l
            Number of drones
        U_dc
            U_+^d (defender utility when defender successfully protects target)
        U_du
            U_-^d (defender utility when defender fails to protect target)
        U_ac
            U_+^a (attacker utility when defender successfully protects target)
        U_au
            U_-^a (attacker utility when defender fails to protect target)
            
    returns::
        Initial strategies (list of multiple strategies, which are each
        a list of sets)
    
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
    Gdir = nx.to_directed(G)
    nodes = set(G.nodes())	
    edges = list(Gdir.edges())
    remainingTarget = set(G.nodes())
    StrategySet = []
    aSet,chosenList,cSet,dSet,eSet,fSet = set(),set(),set(),set(),set(),set()
    n = len(U_dc)
    
    weight = []
    for e in edges:
        weight.append((U_dc[e[0]]+U_dc[e[1]],e))
    weight.sort(reverse=True)

    while(len(remainingTarget)>0 and len(weight) > 0):
        aSet,chosenList,cSet,dSet,eSet,fSet = set(),set(),set(),set(),set(),set()
        PatrolCount = 0
        while PatrolCount < k and len(weight) > 0:
            e = weight[0][1]
            if e[1] in aSet or e[0] in aSet:
                weight = weight[1:]
                continue
            if e[0] in chosenList or e[1] in chosenList:
                weight = weight[1:]
                continue 
            aSet.add(e[0])
            chosenList.add(e[1])
            remainingTarget -= set([e[0]])
            remainingTarget -= set([e[1]])
            weight = weight[1:]
            PatrolCount+=1
        if len(aSet) < k:
            toAdd = random.sample(nodes - chosenList - aSet, k-len(aSet))
            for i in toAdd:
                aSet.add(i)
                chosenList.add(random.sample(G[i].keys(),1)[0])
                remainingTarget -= aSet
                remainingTarget -= chosenList
            
        adjacentList = []
        for xx in aSet:
            outEdges = G[xx].keys()
            adjacentList += outEdges #all neighbors
        remainingNodes = nodes - aSet
        adjacentSet = set(adjacentList)
        
        droneSet = set(random.sample(nodes-aSet, min(l,len(nodes-aSet))))
        dSet = droneSet - adjacentSet #drone, no patrollers nearby (d)
        emptyNodes = remainingNodes - droneSet #nothing at these nodes
        bSet = emptyNodes.intersection(chosenList) #nothing, patroller matched b
        cSet = emptyNodes - bSet #nothing, no patroller matched (c)
        fSet = droneSet.intersection(chosenList) #drone with patroller matched e
        eSet = droneSet - fSet - dSet #drone, no patroller matched (f)
                

        StrategySet.append([aSet,bSet,cSet,dSet,eSet,fSet])
                      
    return StrategySet


def enumerateAll(G, k, l, MAXIteration=None):
    """
    title::
        enumerateAll

    description::
        Generate all pure strategies. See compare_heuristic_script.py for 
        example use.

    attributes::
        G
            Graph object (networkx)
        k
            Number of human patrollers
        l
            Number of drones
        MAXIteration
            Maximum number of strategies or None if no maximum should be set
            [default is None]
            
    returns::
        All strategies (list of multiple strategies, which are each
        a list of sets)
    
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
    nodes = G.nodes()
    strategies = []
    KStrategies = itertools.combinations(nodes, k)
    for x in KStrategies:
        if MUTE != 1: print('x', x)
        adjacentList = []
        chosenList = []
        for xx in x:
            outEdges = G[xx].keys()
            adjacentList += outEdges #all neighbors
            chosenList.append(outEdges)
        if MUTE != 1: print('adj list', adjacentList)
        adjacentSet = set(adjacentList)
        if MUTE != 1: print('adj set', adjacentSet)
        adjacentList = list(adjacentSet)
        remainingNodes = list(set(nodes) - set(x))
        if MUTE != 1: print(x, remainingNodes)
        possibleVisits = list(itertools.product(*chosenList))
        if MUTE != 1: print('possibleVisits', possibleVisits)
        LStrategies = list(itertools.combinations(remainingNodes, l))
        for y in LStrategies:
            for w in possibleVisits:
                if MUTE != 1: print('w,y',w,y)
                dSet = set(y) - adjacentSet #drone, no patrollers nearby (d)
                if MUTE != 1: print('dSet', dSet)
                emptyNodes = set(remainingNodes) - set(y) #nothing
                bSet = emptyNodes.intersection(w) #nothing, patroller match (b)
                if MUTE != 1: print('empty', emptyNodes)
                if MUTE != 1: print('bSet', bSet)
                if MUTE != 1: print('curr chosen',w)
                cSet = emptyNodes - bSet #nothing, no patroller matched (c)
                if MUTE != 1: print('cSet', cSet)
                fSet = set(y).intersection(w) #drone with patroller (e)
                if MUTE != 1: print('fSet', fSet)
                eSet = set(y) - fSet - dSet #drone, no patroller matched (f)
                if MUTE != 1: print('eSet', eSet)
                strategies.append([x, bSet, cSet, dSet, eSet, fSet])
                if MAXIteration != None and len(strategies) > MAXIteration:
                    return strategies
    return strategies