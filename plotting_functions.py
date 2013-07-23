import numpy
import pylab
import matplotlib.pyplot as plt


def plot_ts(ts,taxis,fig_num,title):
    plt.figure(fig_num)
    plt.xlabel('time')
    plt.title('%s' % title)
    plt.plot(taxis[:],ts)


def plot_3plts(ts1,ts2,ts3,taxis,title1,title2,title3,xlab,ylab):
    plt.subplot(311)
#    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title1)
    plt.plot(taxis[:],ts1,'r')
    plt.subplot(312)
#    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title2)
    plt.plot(taxis[:],ts2,'b')
    plt.subplot(313)
    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title3)
    plt.plot(taxis[:],ts3)
    return


