from scipy.special import comb

def maxIterationComb(N,k,l):
    """
    title::
        maxIterationComb

    description::
        Compute N!/k!l!(N-k-l)! (max iterations).

    attributes::
        N
            Number of targets (graph size)
        k
            Number of human patrollers
        l
            Number of drones
    
    returns::
        Resulting maximum iterations (integer).
    
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
    return int(comb(N,k)*comb(N-k,l))