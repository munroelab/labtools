import numpy
import pylab
import matplotlib.pyplot as plt


def plot_ts(ts,taxis,title,xlab,ylab):
    plt.figure()
    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' % title)
    plt.plot(taxis[:],ts)
    return

def sharexy_plot_ts(ts1,ts2,taxis,title,xlab,ylab):
    plt.figure()
    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' % title)
    plt.plot(taxis[:],ts1,taxis[:],ts2)
    return

def sharexy_plot_6plts(ts1,ts2,ts3,ma1,ma2,ma3,taxis,title1,title2,title3,xlab,ylab):
    ax1 = plt.subplot(3,1,1)
    
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title1)
    plt.plot(taxis[:],ts1,'r',taxis[:],ma1)

    plt.subplot(3,1,2,sharex = ax1,sharey=ax1)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title2)
    plt.plot(taxis[:],ts2,'b',taxis[:],ma2)


    pylab.subplot(3,1,3,sharex=ax1,sharey=ax1)
    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title3)
    plt.plot(taxis[:],ts3,taxis[:],ma3)
    return

def sharexy_overlay3plots(ts1,ts2,ts3,time,max_min,var1,var2,var3,xlab,ylab,plotName):
    fig,ax = plt.subplots(1)
    ax.plot(time,ts1,label=var1,color='black')
    ax.plot(time,ts2,label=var2,color='blue')
    ax.plot(time,ts3,label=var3,color='green')
    ax.legend(loc='upper right')
    plt.ylim(0,max_min)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(plotName)
    return


def sharexy_plot_3plts(ts1,ts2,ts3,taxis,title1,title2,title3,xlab,ylab,savefig=False,plotName=None):
    ax1 = plt.subplot(3,1,1)
    
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title1)
    plt.plot(taxis[:],ts1,'r')
    plt.subplot(3,1,2,sharex = ax1,sharey=ax1)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title2)
    plt.plot(taxis[:],ts2,'b')
    pylab.subplot(3,1,3,sharex=ax1,sharey=ax1)
    plt.xlabel('%s' %xlab)
    plt.ylabel('%s' %ylab)
    plt.title('%s' %title3)
    plt.plot(taxis[:],ts3)
    if savefig is True:
        plt.savefig(plotName)
    return

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

