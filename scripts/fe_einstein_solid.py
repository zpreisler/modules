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

    plot(e.l,e.ein.min,"k-",alpha=0.3)
    plot(e.l,e.ein.max,"k-",alpha=0.3)
    errorbar(e.l,e.ein.mean,yerr=sqrt(e.ein.var),fmt='.')

    plot(e.l,e.ein_rot.min,"k-",alpha=0.3)
    plot(e.l,e.ein_rot.max,"k-",alpha=0.3)
    errorbar(e.l,e.ein_rot.mean,yerr=sqrt(e.ein_rot.var),fmt='.')

    figure()
    plot(e.l,e.ein2.min,"k-",alpha=0.3)
    plot(e.l,e.ein2.max,"k-",alpha=0.3)
    errorbar(e.l,e.ein2.mean,yerr=sqrt(e.ein2.var),fmt='.')

    plot(e.l,e.ein_rot2.min,"k-",alpha=0.3)
    plot(e.l,e.ein_rot2.max,"k-",alpha=0.3)
    errorbar(e.l,e.ein_rot2.mean,yerr=sqrt(e.ein_rot2.var),fmt='.')

    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
