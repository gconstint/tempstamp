import h5py
import pylab as plt
import numpy as np
import matplotlib

plt.style.use("ggplot")
pvname1="SBP:FE:GMD1_Ion:Curr:AI"
#pvname1="SBP:EPS:XPi1:ChnB:AI"
pvname2="mso:Area"
pvname3="mso:TimeStamp"




filename= r'E:\scripts_for_commissioning\timestamp_test\2023-05-13\run22\data.hdf5'
f=h5py.File(filename,'r')

##
##plt.scatter(np.arange(0,2000-1),np.abs(np.diff(np.array(f[pvname1]))))
##plt.show()
##
##f=h5py.File(filename,'r')
##diff=np.average(f[pvname1+".timestamp"][:]-f[pvname2+".timestamp"][:])
##for dt in np.arange(-0.2,0.45,0.05):
##    new=f[pvname1+".timestamp"][:]+dt
##    print(dt)
##
##    def findnear(ts1,ts2):
##        pair=[]
##        for i in range(ts1.shape[0]):
##            index=np.argmin(np.abs(ts2-ts1[i]))
##            if np.min(np.abs(ts2-ts1[i]))<0.05:
##                pair.append([i,index])
##        return pair
##    xx=findnear(new,f[pvname2+".timestamp"][:])
##        
##    xx=np.array(xx)
##    print(xx.shape)
##    plt.scatter(np.array(f[pvname1])[xx[:,0]],np.array(f[pvname2])[xx[:,1]])
##    #plt.plot(np.diff(f[pvname3]),label=pvname3)
##    plt.legend()
##    plt.show()




p1=np.polyfit(np.arange(np.shape(f[pvname1+".timestamp"])[0]),f[pvname1+".timestamp"],1)
p2=np.polyfit(np.arange(np.shape(f[pvname2+".timestamp"])[0]),f[pvname2+".timestamp"],1)
fig=plt.figure(figsize=(15,8),dpi=150)

ax1=fig.add_subplot(2,3,1)
ax1.scatter(f[pvname1],f[pvname2])
ax1.set_xlabel(pvname1)
ax1.set_ylabel(pvname2)
ax1.set_title("correlation@0")

ax2=fig.add_subplot(2,3,2)
ax2.scatter(np.array(f[pvname1])[:-1],np.array(f[pvname2])[1:])
ax2.set_xlabel(pvname1)
ax2.set_ylabel(pvname2)
ax2.set_title("correlation@1")
ax3=fig.add_subplot(2,3,3)
ax3.scatter(np.array(f[pvname1])[1:],np.array(f[pvname2])[:-1])
ax3.set_xlabel(pvname1)
ax3.set_ylabel(pvname2)
ax3.set_title("correlation@-1")

ax4=fig.add_subplot(2,3,4)
ax4.plot(np.diff(f[pvname1+".timestamp"]),label=pvname1+".timestamp")
ax4.plot(np.diff(f[pvname2+".timestamp"]),label=pvname2+".timestamp")
ax4.set_xlabel("pulse number")
ax4.set_ylabel("interval")
ax4.set_title("interval of timestamp")
ax4.legend()

ax5=fig.add_subplot(2,3,5)
ax5.scatter(np.arange(0,np.array(f[pvname1]).shape[0]),np.array(f[pvname1]),s=5,label="timestamp difference")
ax5.set_xlabel("pulse number")
ax5.set_ylabel("difference")
ax5.set_title("difference between\n %s \n and %s"%(pvname2,pvname1))
ax5.legend()

ax6=fig.add_subplot(2,3,6)
ax6.plot(f[pvname1+".timestamp"]-np.arange(np.shape(f[pvname1+".timestamp"])[0])*p1[0]+p1[1],label=pvname1+".timestamp")
ax6.plot(f[pvname2+".timestamp"]-np.arange(np.shape(f[pvname2+".timestamp"])[0])*p2[0]+p2[1],label=pvname2+".timestamp")
ax6.set_xlabel("pulse number")
ax6.set_ylabel("nonlinear")
ax6.set_title("nonlinear of timestamp")
ax6.legend()
plt.tight_layout()
fig.savefig(filename+".png",dpi=600)

plt.show()
