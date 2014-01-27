
import numpy as np
import argparse
import pylab


def predictions_wave(N,time,omega=0):
    #kz is constant
    kz = 0.063
    N=N*1.0
    time = time*1.0
    
    omega=2.0*np.pi/time
    
    theta = np.arccos(omega/N)
    theta_deg = theta * 180/ np.pi
    kx = (omega * kz) / ((N*N - omega*omega)**0.5)
    c_gx = N*np.cos(theta)*(np.sin(theta)**2)/kx
    print "******Predictions********"
    print "theta : %f" %theta_deg
    print "omega : %f " %omega
    print "kx : %f " %kx
    print "wavelength : %f m " %(2*np.pi/kx)
    print "wave speed Cgx : %f cm/s " %c_gx

    # wave generator settings
    setting_4cm = 60 * 200 / time
    settings1 = setting_4cm / 3.2
    settings2 = setting_4cm / 4
    print "***** WAVE GENERATOR SETTINGS *****"
    print "RPM (4cm Amplitude) : %d" %setting_4cm
    print "RPM (2.5cm Amplitude) : %d" %settings1
    print "RPM (2cm Amplitude) : %d" %settings2
    return


def UI():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("N",type=float,help="buoyancy frequency")
    parser.add_argument("time",type = float,help= "time period of the wave")
    parser.add_argument("--omega", type=float,help="wave frequency")

    args = parser.parse_args()
    predictions_wave(args.N,args.time,args.omega)

if __name__ == "__main__":
    UI()