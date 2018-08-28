class MOF_lattice:
    """
    MOF class
    """
    del_k_m=0.0001
    del_k_s=0.0001
    del_k_t=0.0001
    def __init__(self,N_sys,E_s,E_m,E_t):
        self.N_sys=N_sys
        self.E_s=E_s
        self.E_m=E_m
        self.E_t=E_t

    def get_weights(self,mu,kT):
        from numpy import exp
        """
        Returns the statistical weights at a given reduced chemical potential and reduced temperature.
        """
        K_s=exp((mu-self.E_s)/kT)
        K_m=exp((mu-self.E_m)/kT)
        K_t=exp((mu-self.E_t)/kT)
        return K_s,K_m,K_t

    def logpartition_function(self,K_s_p,K_m_p,K_t_p,N_s):
        from numpy import sqrt,log,power
        """
        Returns (1/N) ln Z  at a given reduced chemical potential and reduced temperature.
        """
        denom=sqrt((1+K_s_p-K_m_p)**2+(4.0*K_t_p*K_t_p))
        eval_0=0.5*((1+K_s_p+K_m_p)+denom)
        eval_1=0.5*((1+K_s_p+K_m_p)-denom)
        
        term_1=log(((1+K_s_p-eval_1)+((power((eval_1/eval_0),N_s))*(eval_0-1-K_s_p)))/(eval_0-eval_1))
        return(term_1/N_s)+(log(eval_0))  

    def loading_species(self,mu,kT):
        from numpy import array
        """
        Returns the total uptake or loading at a given reduced chemical potential and reduced temperature.
        """
        a=array([*self.get_weights(mu,kT)+(self.N_sys,)])
        s=a+[self.del_k_s,.0,.0,0]
        m=a+[0.,self.del_k_s,.0,0]
        t=a+[0.,0.,self.del_k_s,0]
        K_s,K_m,K_t,N=a

        logZ=self.logpartition_function(*a)
        logZs=self.logpartition_function(*s)
        logZm=self.logpartition_function(*m)
        logZt=self.logpartition_function(*t)

        der_s=(K_s/self.del_k_s)*(logZs-logZ)
        der_m=(K_m/self.del_k_m)*(logZm-logZ)
        der_t=(K_t/self.del_k_t)*(logZt-logZ)
 
        return der_s,der_m,der_t

    def loading(self,mu,kT):
        """
        Returns the uptake or loading of every species at a given reduced chemical potential and reduced temperature.
        """
        return sum(self.loading_species(mu,kT))

    def get_loading(self,mu_list,temp_list):
        """
        Returns the loading for given chemical potentials and reduced temperatures.
        """
        return [self.loading(mu,temp)for mu,temp in zip(mu_list,temp_list)]

    def free_energy(self,mu,kT):
        """
        F/KT - Dimensionless
        """
        return kT*self.N_sys*self.logpartition_function(*get_weights(mu,kT)+(self.N_sys,))

    def hessian(self,mu,kT):
        from numpy import zeros
        """
        Returns the Hessian matrix at a given reduced chemical potential and reduced temperature.
        """
        rho_s,rho_m,rho_t=self.loading_species(mu,kT)

        hess_mat=zeros((3,3))

        hess_mat[0,0]=(1.0/rho_s) + (1.0/(1.0-rho_s-rho_m-rho_t))
        hess_mat[1,0]=(1.0/(1.0-rho_s-rho_m-rho_t))
        hess_mat[2,0]=(1.0/(1.0-rho_s-rho_m-rho_t))

        hess_mat[0,1]=(1.0/(1.0-rho_s-rho_m-rho_t))
        hess_mat[1,1]=(-1.0/(1-rho_m-(0.5*rho_t))) - (1.0/(rho_m+(0.50*rho_t))) + (1.0/rho_m)  + (1.0/(1.0-rho_s-rho_m-rho_t)) 
        hess_mat[2,1]=(-0.5/(1-rho_m-(0.5*rho_t))) - (0.5/(rho_m+(0.50*rho_t)))                + (1.0/(1.0-rho_s-rho_m-rho_t)) 

        hess_mat[0,2]=(1.0/(1.0-rho_s-rho_m-rho_t))
        hess_mat[1,2]=(-0.5/(1-rho_m-(0.5*rho_t))) - (0.5/(rho_m+(0.50*rho_t)))                 + (1.0/(1.0-rho_s-rho_m-rho_t)) 
        hess_mat[2,2]=(-0.25/(1-rho_m-(0.5*rho_t))) - (0.25/(rho_m+(0.50*rho_t))) + (1.0/rho_t) + (1.0/(1.0-rho_s-rho_m-rho_t)) 

        return hess_mat

    def free_energy_inf_app(self,mu,kT):
        from numpy import log
        """
        F/NkT -- infinite system approximation
        """
        rho_s,rho_m,rho_t=self.loading_species(mu,kT)
        rho_t2=0.5*rho_t
        fe=(
                (-(1.0-rho_m-rho_t2)*log(1.0-rho_m-rho_t2))-
                ((rho_m+rho_t2)*log(rho_m + rho_t2))+
                (rho_s*log(rho_s))+
                (rho_m*log(rho_m))+
                (rho_t*math.log(rho_t2))+
                (1.0-rho_s-rho_m-rho_t)*log(1.0-rho_s-rho_m-rho_t))
        return fe
    
    def chain_length_dist(self,mu,kT):
        from numpy import zeros,log,exp,arange
        """
        Output is l, r(l) where r(l) is the probability of finding a chain of length l
        """
        rho_s,rho_m,rho_t=self.loading_species(mu,kT)

        l0=-1.0/log((2.0*rho_m)/(rho_t+(2.0*rho_m)))
        rl=zeros(self.N_sys/2)
        rl=[(rho_t/(rho_t+(2.0*rho_m)))*exp(-(li-2)/l0) for li in range(self.N_sys/2)]

        return arange(self.N_sys/2), rl
    
    def correlation_length(self,mu,kT):
        from numpy import sqrt,log
        """
        Bond-bond correlation length of the system
        """
        K_s,K_m,K_t=self.get_weights(mu,kT)
        
        denom=sqrt((1+K_s-K_m)**2+(4.0*K_t*K_t))
        eval_0=0.5*((1+K_s+K_m)+denom )
        eval_1=0.5*((1+K_s+K_m)-denom )
        
        eps_length=-1.0/log(eval_1/eval_0)
        return eps_length
