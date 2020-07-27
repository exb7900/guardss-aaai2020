import numpy as np

def calculate_eu(target,
                 gamma,
                 eta,
                 uMat,
                 npvar,
                 N,
                 U_dc,
                 U_du,
                 U_ac,
                 U_au,
                 mode,
                 computeMode='full'):

    """
    title::
        calculate_eu

    description::
        Objective function in Appendix LP (Equation 1), for attacker or
        defender. This provides the expected utility.

    attributes::
        target
            Current target for which to solve
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
        npvar
            Final optimal decision variable values, make sure to input as numpy
            array, e.g., np.array(variables)
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
        mode
            Whether to calculate "attacker" or "defender" expected utility
        computeMode
            Whether to calculate "full" (meaning Appendix LP) or "detection" 
            only (meaning in main paper). Note that "detection" is depracated,
            and was not used for experiments, but is included here to provide
            some guidance just in case.
    
    returns::
        Defender or attacker expected utility
    
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
    if mode == 'attacker':
        U_c = U_ac
        U_u = U_au
    else:
        U_c = U_dc
        U_u = U_du

    objectiveValueCoef = np.zeros(len(npvar))
    if computeMode == 'full':
        uQuiet = (uMat[0,1]*eta[0] + uMat[1,1]*eta[1] + uMat[2,1]*eta[2])
        uSignal = (uMat[0,2]*eta[0] + uMat[1,2]*eta[1] + uMat[2,2]*eta[2])

        #U_-s^a/d
        objectiveValueCoef[0*N + target] = U_c[target] #x_t^p * U_c
        objectiveValueCoef[1*N + target] = U_c[target] * eta[0] #x_t^n+ * U_c
        objectiveValueCoef[2*N + target] = U_u[target] * eta[0] #x_t^n- * U_u

        #Combination of U_sigma1^a/d and U_sigma0^a/d.
        #x's in U_sigma1^a/d
        objectiveValueCoef[5*N + target] = U_c[target] * uSignal
        objectiveValueCoef[4*N + target] = ((1-gamma) * U_c[target] * \
                                            uSignal) + \
                                            (gamma * U_u[target] * uSignal)
        objectiveValueCoef[3*N + target] = U_u[target] * uSignal
        #U_c psi_t^s+
        objectiveValueCoef[8*N + target] = ((1-gamma) * U_c[target] * \
                                            uQuiet) - \
                                            ((1-gamma) * U_c[target] * \
                                            uSignal)
        #U_c psi_t^s-
        objectiveValueCoef[6*N + target] = ((1-gamma) * U_c[target] * \
                                            uQuiet) - \
                                            ((1-gamma) * U_c[target] * \
                                            uSignal)
        #U_u psi_t^sbar
        objectiveValueCoef[7*N + target] = ((1-gamma) * U_u[target] * \
                                            uQuiet) - \
                                            ((1-gamma) * U_u[target] * \
                                            uSignal)
        #U_c phi_t^s+
        objectiveValueCoef[11*N + target] = (gamma * U_c[target] * \
                                                uQuiet) - \
                                            (gamma * U_c[target] * uSignal)
        #U_u phi_t^s-            
        objectiveValueCoef[9*N + target] = (gamma * U_u[target] * uQuiet)-\
                                            (gamma * U_u[target] * uSignal)
        #U_u phi_t^sbar
        objectiveValueCoef[10*N + target] = (gamma * U_u[target] * \
                                                uQuiet) - \
                                            (gamma * U_u[target] * uSignal)
    else:
        objectiveValueCoef[0*N + target] = U_c[target] #x_t^p * U_c
        objectiveValueCoef[1*N + target] = U_c[target] #x_t^n+ * U_c
        objectiveValueCoef[2*N + target] = U_u[target] #x_t^n- * U_u
        objectiveValueCoef[8*N + target] = (1-gamma) * U_c[target] #U_c psi_t^s+
        objectiveValueCoef[6*N + target] = (1-gamma) * U_c[target] #U_c psi_t^s-
        objectiveValueCoef[7*N + target] = (1-gamma) * U_u[target] #U_u psi_t^sbar
        objectiveValueCoef[11*N + target] = gamma * U_c[target] #U_c phi_t^s+
        objectiveValueCoef[9*N + target] = gamma * U_u[target] #U_u phi_t^s-
        objectiveValueCoef[10*N + target] = gamma * U_u[target] #U_u phi_t^sbar

    return objectiveValueCoef.dot(npvar)



def compute_real_optimal_attacker(N,
                                  variables,
                                  strategies,
                                  G,
                                  uMat,
                                  gamma,
                                  U_dc,
                                  U_du,
                                  U_ac,
                                  U_au,
                                  computeMode='full'):

    """
    title::
        compute_real_optimal_attacker

    description::
        Recalculate objective value for variable assignments from run without 
        considering uncertainty, but with the true uncertainty matrix/gamma.

    attributes::
        N
            Number of targets (graph size)
        variables
            Final optimal decision variable values
        strategies
            Pure strategies (None if wanting to run relaxed version)
        G
            Graph object (networkx)
        uMat
            Uncertainty matrix \Pi will contain the conditional probability 
            Pr[\omega^|\omega] for all \omega^, \omega \in \Omega
            to describe how likely the attacker will observe a signaling
            state \omega^ given the true signaling state is \omega
        gamma
            False negative rate
        U_dc
            U_+^d (defender utility when defender successfully protects target)
        U_du
            U_-^d (defender utility when defender fails to protect target)
        U_ac
            U_+^a (attacker utility when defender successfully protects target)
        U_au
            U_-^a (attacker utility when defender fails to protect target)
        computeMode
            Whether to calculate "full" (meaning Appendix LP) or "detection" 
            only (meaning in main paper). Note that "detection" is depracated,
            and was not used for experiments, but is included here to provide
            some guidance just in case.

    returns::
        curObj
            Actual defender expected utility with uncertainty
        bestTarget
            Attacker best target in this uncertainty scenario
        bestEta
            Attacker best behavior in this uncertainty scenario

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
    npvar = np.array(variables)

    objValues = []
    allTs = []
    allEtas = []

    for target in range(N):
        if computeMode == 'full':
            etais = [[0,0,0]]
            etai = [0,0,0]

            #Determine relevant attacker behaviors (without looping through all
            #possible etas, to save time). Based on Appendix Equation 10.
            for state in range(3):
                UTheta = np.zeros(len(variables))

                UTheta[1*N+target] = U_ac[target]*uMat[state,0]
                UTheta[2*N+target] = U_au[target]*uMat[state,0]
                UTheta[5*N+target] = U_ac[target]*uMat[state,2]
                UTheta[4*N+target] = ((1-gamma) * U_ac[target]*uMat[state,2])+\
                                    (gamma*U_au[target]*uMat[state,2])
                UTheta[3*N+target] = U_au[target]*uMat[state,2]
                UTheta[8*N+target] = ((1-gamma) * -U_ac[target] * uMat[state,2])+\
                                    ((1-gamma) * U_ac[target] * uMat[state,1])
                UTheta[6*N+target] = ((1-gamma) * -U_ac[target] * uMat[state,2])+\
                                    ((1-gamma) * U_ac[target] * uMat[state,1])
                UTheta[7*N+target] = ((1-gamma) * -U_au[target] * uMat[state,2])+\
                                    ((1-gamma) * U_au[target] * uMat[state,1])
                UTheta[11*N+target] = (gamma * -U_ac[target] * uMat[state,2])+\
                                    (gamma * U_ac[target] * uMat[state,1])
                UTheta[9*N+target] = (gamma * -U_au[target] * uMat[state,2])+\
                                    (gamma * U_au[target] * uMat[state,1])
                UTheta[10*N+target] = (gamma * -U_au[target] * uMat[state,2])+\
                                    (gamma * U_au[target] * uMat[state,1])

                #If it's positive, eta should be 1.
                if UTheta.dot(npvar) > 1e-3:
                    for etai in etais:
                        etai[state] = 1
                #If it's negative, eta should be 0.
                elif UTheta.dot(npvar) < -1e-3:
                    for etai in etais:
                        etai[state] = 0
                #If it's zero, could be 0 or 1, so add both.
                else:
                    addEtais =[]
                    for etai in etais:
                        etai[state] = 0
                        addEtais.append(etai.copy())
                        addEtais[-1][state] = 1
                    etais += addEtais


            #Loop through attacker behaviors and calculate attacker expected 
            #utilities.
            for etai in etais:
                attEU = calculate_eu(target,
                                    gamma,
                                    etai,
                                    uMat,
                                    npvar,
                                    N,
                                    U_dc,
                                    U_du,
                                    U_ac,
                                    U_au,
                                    mode='attacker',
                                    computeMode=computeMode)

                objValues.append(attEU)
                allTs.append(target)
                allEtas.append(etai)
        else:
            #Don't need behaviors for loop, just calculate directly.
            attEU = calculate_eu(target,
                                 gamma,
                                 etai,
                                 uMat,
                                 npvar,
                                 N,
                                 U_dc,
                                 U_du,
                                 U_ac,
                                 U_au,
                                 mode='attacker',
                                 computeMode=computeMode)
            objValues.append(attEU)
            allTs.append(target)

    #Determine best attacker responses (there could be ties).
    objValues = np.array(objValues)
    attackerResponseTies = np.where(objValues >= objValues.max()-1e-5)

    allTs = np.array(allTs)
    allEtas = np.array(allEtas)

    bestTargets = allTs[attackerResponseTies]
    objValuesBest = objValues[attackerResponseTies]
    if computeMode == 'full':
        bestEtas = allEtas[attackerResponseTies]

    curObj = -float('inf')
    curTarget = None
    curEta = None

    #Break ties in favor of defender by computing defender expected utilities.
    for SOMEITERATION in range(len(bestTargets)):
        if computeMode == 'full':
            bestEta = bestEtas[SOMEITERATION]
        else:
            bestEta = None
        bestTarget = bestTargets[SOMEITERATION]
        defEU = calculate_eu(bestTarget, 
                             gamma, 
                             bestEta,
                             uMat,
                             npvar,
                             N,
                             U_dc,
                             U_du,
                             U_ac,
                             U_au,
                             mode='defender',
                             computeMode=computeMode)
        
        if curObj < defEU:
            curObj = defEU
            curTarget = bestTarget
            curEta = bestEta

    bestTarget = curTarget
    bestEta = curEta

    return curObj, bestTarget, bestEta