#!/usr/bin/env python
def main():
    from myutils import configuration
    from myutils.einstein import einstein
    from argparse import ArgumentParser
    from numpy import log,exp,savetxt,array
    p=ArgumentParser()
    p.add_argument('files',nargs='+')

    files=p.parse_args().files
    e=einstein(files)

    fe_trans=e.gauss_integrate('trans')
    fe_rot=e.gauss_integrate('rot')

    fe_einstein=e.einstein_solid_fe()
    fe_id_rot=e.rot_fe()
    fe_energy=e.energy_fe()


    fe_all=fe_trans+fe_rot+fe_einstein+fe_id_rot+fe_energy
    savetxt("fe.out",(e.beta,fe_all))

    f="""
    Einstein solid free energy calculation
    ######################################

    translational free energy:  % .3lf
    rotational free energy:     % .3lf
    Einstein solid free energy: % .3lf
    reference rotational fe:    % .3lf
    energy contribution:        % .3lf
                                -------    
    beta: %.3lf
    beta f:                     % .3lf
    """%(fe_trans,fe_rot,fe_einstein,fe_id_rot,fe_energy,e.beta,fe_all)

    print(f)

    from matplotlib.pyplot import show,plot,errorbar,figure,savefig

    figure()
    e.plot('en',label=r"$\left<u\right>$")

    figure()
    e.plot('ein')
    e.plot('ein_rot',label=r"$\left< H\right>$")
    savefig("ein.pdf")
    savefig("ein.png")

    figure()
    e.plot('ein2')
    e.plot('ein_rot2')
    savefig("ein2.pdf")
    savefig("ein2.png")

    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
