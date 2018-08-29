#!/usr/bin/env python
def main():
    """
    Plot
    """
    from matplotlib.pyplot import figure,show,plot,errorbar,subplots,xlabel,ylabel,subplots_adjust,savefig
    from numpy import loadtxt,arange,array,sqrt,linspace
    from mof_lattice import MOF_lattice,MOF_data,MOF

    n=100
    f=MOF()

    mof=MOF_lattice(f.length,f.E_s,f.E_m,f.E_t)
    temp=[f.temperature]*n
    mu=linspace(f.mu.min()-0.1,f.mu.max()+0.1,n)
    rho=mof.get_loading(mu,temp)

    fig,ax1=subplots()

    ax1.plot(mu,rho,'g-',linewidth=1.0)

    errorbar(f.mu,f.rho.mean,yerr=sqrt(f.rho.var),linewidth=1.00,elinewidth=6.00,ecolor="y",barsabove=True,color="r",alpha=0.66)
    ax1.vlines(f.mu,ymin=f.rho.min,ymax=f.rho.max,linewidth=0.66,alpha=1.0)
    ax1.plot(f.mu,f.rho.mean,"ro",markersize=1.0,alpha=0.66)
    xlabel(r"$\mu$")
    ylabel(r"$\rho$")
    subplots_adjust(left=0.18,bottom=0.18)
    savefig("mof.png")
    savefig("mof.pdf")

    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
