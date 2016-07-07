"""
Routine to aid in creation of stratifications
"""
from nacl_solution import *
import pylab
import numpy

# There is a relationship (Weast 1981) given between the concentation (mass
# solute / volume solution) and the density of the solution
# e.g. Cs_to_rho() and rho_to_Cs()

# NaCl is saturated at 26%
# double bucket apparatus:
# C(z) = C0 + (C1 - C0) * A * h / (2 * V0)
# C1 - concentraion of salt bucket
# C0 - initial concentration of mixing bucket
# V0 - initial volume of mixing bucket
# A - cross sectional area of tank to be filled

# constants
rho_fw = 0.9982 # density of freshwater
g = 980.0

# Formula for concentation filling from the top
# C(z) = C0 + (C1 - C0) * A * z / (2 * V0)

def estimate_N(A, H, rho1, rho0, V0):
    """
    Compute the expected value of N given the the parameters:
        rho1 = density of "added water" bucket
        rho0 = initial density of "mixing bucket"
        V0 = initial volume of "mixing bucket"
    based on filling a tank of cross sectional area A
    when filling the tank to a depth of H
    """

    # note: Cs_to_rho is *not* linear!
    # which is why we are going this this 'polyfit' business

    rho_fw = 0.9982

    C0 = rho_to_Cs(rho0)
    C1 = rho_to_Cs(rho1)

    rhos = []
    Z = numpy.mgrid[0:H:20j]
    for z in Z:
        C = C0 + (C1-C0) * A * (z) / (2 * V0)
        rho = Cs_to_rho(C)
        rhos.append(rho)

    rhos = pylab.array(rhos)
    pylab.plot(rhos, Z)

    # fit a straight line to the profile
    drhodz, rhoB = pylab.polyfit(Z, rhos, 1)

    N2 = - g / rho_fw * drhodz
    N = pylab.sqrt(N2)
    return N

def plot():
    """
    Make a plot of all stratifications possible for given 
    tank dimensions
    """

    # Tank parameters:
    H = 52.0  # important parameter??
    L = 488.0 
    W = 46.0
    A = L*W

    rho0 = 0.9982

    # assume these paramters are variables:
    # parameters C1, V0
    V0 = pylab.arange(0, 1000e3, 20e3)
    V0 = numpy.mgrid[0:1000e3:50j]
    rho1 = numpy.mgrid[rho0:1.15:50j]

    N = pylab.zeros((len(rho1), len(V0)))

    for i, V0_ in enumerate(V0):
        for j, rho1_ in enumerate(rho1):
            N[j, i] = estimate_N(A, H, rho1_, V0_, rho0=rho0)

    V0, rho1 = pylab.meshgrid(V0, rho1)

    pylab.contourf(V0/1000, rho1, N, pylab.arange(0, 3.0, 0.1))
    pylab.colorbar()

    pylab.xlabel(r'Initial Volume of Mixing Tank $V_0$')
    pylab.ylabel(r'Density of SW Tank $\rho_{sw}$')
    pylab.axis('tight')
    pylab.title('L = %.1f, W = %.1f, H= %.1f' % (L, W, H))

    # minimum volume in FW bucket needed to fill the tank
    V0_min = 0.5 * A*H

    # minimum volume needed for mixer to function
    #A_fw = 60*60
    #V0_mixer = A_fw*4
    V0_mixer = 200e3 # L

    pylab.axvline(V0_min/1000, color='k')
    pylab.axvline((V0_min+V0_mixer)/1000, color='k',linestyle='--')


def other_tests():

    rho0 = 0.9982

    # Tank parameters:
    H = 15.9  # important parameter??
    L = 51.2 
    W = 17.3
    A = L*W

    # assume these paramters are variables:
    # parameters C1, V0
    V0 = numpy.mgrid[10000:80000:50j]
    rho1 = numpy.mgrid[rho0+0.001:1.15:50j]

    N = pylab.zeros((len(rho1), len(V0)))

    for i, V0_ in enumerate(V0):
        for j, rho1_ in enumerate(rho1):
            N[j, i] = estimate_N(A, H, rho1_, V0_, rho0=rho0)


    print estimate_N(A, H, rho0, 10000)

    #V0, rho1 = pylab.meshgrid(V0, rho1)

    # minimum volume in FW bucket needed to fill the tank
    V0_min = 0.5 * A*H

    # minimum volume needed for mixer to function
    A_fw = 60*60
    V0_mixer = A_fw*4

    # what is the minimum volume on the FW bucket?
    i = abs((V0 - (V0_min + V0_mixer))).argmin()
    V0_rec = V0[i]

    Ngoal = 1.0

    #what density to achieve desired N given V0_rec?
    print N[:,i] - Ngoal

    j = abs(N[:,i] - Ngoal).argmin()
    rho1_rec = rho1[j]

    print 'To achieve a buoyancy of N = %.2f' % N[j, i]
    print 'Use V0 = %.0f L' % (V0_rec/1000), ' and rho1 = %.3f' % rho1_rec
    print 'Fill the FW bucket to a depth of h = %.1f cm' % (V0_rec / A_fw)
    print

    V_sw = V0_min + A_fw*2.0
    print 'Prepare a SW solution of volume V_sw = %.0f L' % V_sw
    print 'Cs =', rho_to_Cs(rho1_rec)
    print 'Cw =', rho_to_Cw(rho1_rec)
    print 'Using M_NaCl = %.0f g ' % (rho_to_Cs(rho1_rec)*V_sw)
    V_h20_sw = rho_to_Cw(rho1_rec)*V_sw / rho_fw
    print 'mixed into a Volume of H20 = %.1f L' % (V_h20_sw / 1000)
    print ' (height of water h = %.1f cm)' % (V_h20_sw / A_fw)

    V0, rho1 = pylab.meshgrid(V0, rho1)

    pylab.contourf(V0/1000, rho1, N, pylab.arange(0.0, 3.0, 0.1))
    pylab.colorbar()

    pylab.xlabel(r'Volume of FW Tank $V_0$')
    pylab.ylabel(r'Density of SW Tank $\rho_{sw}$')
    pylab.axis('tight')
    pylab.title('L = %.1f, W = %.1f, H= %.1f' % (L, W, H))

    pylab.plot(V0_rec/1000, rho1_rec, 'wo', mec='k')

    pylab.axvline(V0_min/1000, color='k')
    pylab.axvline((V0_min+V0_mixer)/1000, color='k',linestyle='--')


    ##
    print "-"*40

    # current density of salt water
    rho_sw = 1.041
    # current volume of salt water
    h_sw = 9.5 # cm
    V_sw = A_fw*h_sw

    print "Have a density of rho_sw = %.3f" % rho_sw
    print "Have a volume of V_sw = %.1f L" % (V_sw /1000)

    print "Mass of NaCl = %.0f g" % (rho_to_Cs(rho_sw) * V_sw)
    M_nacl_1 = rho_to_Cs(rho_sw) * V_sw
    print "Mass of H2O = %.0f g" % (rho_to_Cw(rho_sw) * V_sw)
    V_h20_sw = rho_to_Cw(rho_sw) * V_sw / rho_fw
    print "Volume of H2O = %.1f L" % (V_h20_sw / 1000)

    # volume of final solution
    V_sw =  V_sw * rho_to_Cw(rho_sw) / rho_to_Cw(rho1_rec)
    # amount of H2O in final solution (should remain unchanged)
    V_h20_sw = rho_to_Cw(rho1_rec) * V_sw / rho_fw
    print 'Prepare a SW solution of volume V_sw = %.0f L' % V_sw
    print 'Cs =', rho_to_Cs(rho1_rec)
    print 'Cw =', rho_to_Cw(rho1_rec)
    print 'Using M_NaCl = %.0f g ' % (rho_to_Cs(rho1_rec)*V_sw)
    M_nacl_2 = rho_to_Cs(rho1_rec) * V_sw
    print '(Add an additional %.0f g of NaCl)' % ( M_nacl_2 - M_nacl_1)
    print "Mass of H2O = %.0f g" % (rho_to_Cw(rho_sw) * V_sw)
    print 'mixed into a Volume of H20 = %.1f L' % (V_h20_sw / 1000)
    print ' (height of water h = %.1f cm)' % (V_h20_sw / A_fw)

    C0 = rho_to_Cs(rho0) # density of freshwater

    print "Saturated solution"
    # saturated solution
    rho2 = 1.175
    print "Cs = %.3f g" % (rho_to_Cs(rho2))
    M_nacl_add = M_nacl_2 - M_nacl_1
    print "To get enough salt, need %.0f mL of saturated solution" % (M_nacl_add / rho_to_Cs(rho2))


def strat_20100513():

    # Tank parameters:
    H = 15.9  # important parameter??
    L = 63.5 
    W = 17.3
    A = L*W
 
    rho0 = 0.9982

    rho1 = 1.1278

    h0 = 6 
    V0 = 60*60*h0

    pylab.plot(V0/1000, rho1, 'ko')
    print "N =", estimate_N(A, H, rho0=rho0, rho1=rho1, V0=V0)


def strat_20100531():

    # Tank parameters:
    H = 32.0  # important parameter??
    L = 148.0 
    W = 17.3
    A = L*W
 
    rho0 = 0.9982

    rho1 = 1.1278

    h0 = 6 
    V0 = 60*60*h0

    pylab.plot(V0/1000, rho1, 'ko')
    print "N =", estimate_N(A, H, rho0=rho0, rho1=rho1, V0=V0)

def strat_20100715():

    # tank parameters:
    h = 15.9  # important parameter??
    l = 51.2 
    w = 17.3
    a = l*w
 
    rho0 = 0.9982

    rho1 = 1.1278

    h0 = 6 
    v0 = 60*60*h0

    pylab.plot(v0/1000, rho1, 'ko')
    print "n =", estimate_n(a, h, rho0=rho0, rho1=rho1, v0=v0)

def strat_20130311():

    # tank parameters:
    h = 52.0  # important parameter??
    l = 488.0 
    w = 46.0
    a = l*w
 
    # added water density
    rho1 = 1.000

    # mixing bucket initial density
    rho0 = 1.0735
    V0 = 800e3

    #pylab.plot(V0/1000, rho1, 'ko')
    print "n =", estimate_N(a, h, rho1, rho0, V0=V0)

    print "MNacl=", rho_to_Cs(1.053) * 600e3
    MNacl = rho_to_Cs(1.053) * 600e3

    add_salt = 40e3
    print "Adding", add_salt, "gives a new MNacl =",  (MNacl + add_salt)
    print "New concentration:", (MNacl + add_salt) / 800e3
    print "Goal: Cs =", rho_to_Cs(1.0735), "g/mL"

if __name__ == "__main__":
    #plot()

    #strat_20100513()
    #other_tests()

    strat_20130311()

    pylab.show()
