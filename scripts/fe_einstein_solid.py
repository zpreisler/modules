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

    F_trans=e.gauss_integrate('trans')
    F_rot=e.gauss_integrate('rot')

    F_einstein=e.einstein_solid_fe()
    F_id_rot=e.rot_fe()
    F_energy=e.energy_fe()

    f="F_trans %.3lf\nF_rot %.3lf\nF_einstein %.3lf\nF_id_rot %.3lf\nF_energy %.3lf\n"%(F_trans,F_rot,F_einstein,F_id_rot,F_energy)
    print(f)
    F_all=F_trans+F_rot+F_einstein+F_id_rot+F_energy
    print("beta F_all %.3lf %.3lf\n"%(e.beta,F_all))
    savetxt("fe.out",(e.beta,F_all))

    from matplotlib.pyplot import show,plot,errorbar,figure
    from numpy import sqrt

    figure()
    e.plot('en',label=r"$\left<u\right>$")

    figure()
    e.plot('ein')
    e.plot('ein_rot',label=r"$\left< H\right>$")

    figure()
    e.plot('ein2')
    e.plot('ein_rot2')

    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
