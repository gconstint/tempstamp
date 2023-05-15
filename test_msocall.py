import epics
import h5py
import time,os
import numpy as np
import pylab as plt
import datetime
import time
global number1,number2,number3
global DEBUG_FLAG
DEBUG_FLAG=False


folder="E:\\scripts_for_commissioning\\timestamp_test\\"
subfolder=time.strftime("%Y-%m-%d")
dayfolder=os.path.join(folder,subfolder)
if os.path.exists(dayfolder)==False:
    os.mkdir(dayfolder)
run_number=0
for i in range(10000):
    runfolder=os.path.join(dayfolder,"run"+str(run_number))
    if not os.path.exists(runfolder):
        os.mkdir(runfolder)
        break
    else:
        run_number+=1



data_number=2000

pvname1="SBP:FE:GMD1_Ion:Curr:AI"
pvname2="mso:Area"
pvname3="mso:TimeStamp"

number1=0
number2=0
number3=0
filename=os.path.join(runfolder,"data.hdf5")
f=h5py.File(filename,'w')

GMD1=f.create_dataset(pvname1,(data_number,),dtype=np.float64)
GMD1Timestamp=f.create_dataset(pvname1+".timestamp",(data_number,),dtype=np.float64)

msoArea=f.create_dataset(pvname2,(data_number,),dtype=np.float64)
msoTimestamp=f.create_dataset(pvname2+".timestamp",(data_number,),dtype=np.float64)

msoArea_timestamp=f.create_dataset(pvname3,(data_number,),dtype=np.float64)
msoTimestamp_timestamp=f.create_dataset(pvname3+".timestamp",(data_number,),dtype=np.float64)

def time2timestamp(time_str):
    time_obj=datetime.datetime.strptime(time_str,'%d.%m.%Y.%H:%M:%S.%f')
    timestamp=time.mktime(time_obj.timetuple())+(time_obj.microsecond/1e6)
    return timestamp

# def common_data_callback(pvname,value,timestamp,**kwargs):
#     global DEBUG_FLAG
#     global number1
#     if number1<data_number:
#         if DEBUG_FLAG==True:
#             print(value,timestamp,number1)
#         GMD1[number1]=value
#         GMD1Timestamp[number1]=timestamp
#         number1+=1
#     else:
#         PV1.auto_monitor=False
#         print(pvname+" is finished")

def GMD1_data_callback(pvname,value,timestamp,**kwargs):
    global DEBUG_FLAG
    global number1
    if number1<data_number:
        if DEBUG_FLAG==True:
            print(value,timestamp,number1)
        GMD1[number1]=value
        GMD1Timestamp[number1]=timestamp
        number1+=1
    else:
        PV1.auto_monitor=False
        print(pvname+" is finished")

        
def msoArea_data_callback(pvname,value,timestamp,**kwargs):
    global DEBUG_FLAG
    global number2
    if number2<data_number:
        if DEBUG_FLAG==True:
            print(value,timestamp,number2)
        msoArea[number2]=value
        msoTimestamp[number2]=timestamp
        number2+=1
    else:
        PV2.auto_monitor=False
        print(pvname+" is finished")


def msoTimestamp_data_callback(pvname,value,timestamp,**kwargs):
    global DEBUG_FLAG
    global number3
    if number3<data_number:
        if DEBUG_FLAG==True:
            print(value[:-6],timestamp,number3)
        msoTimestamp[number3]=time2timestamp(value[:-6])
        msoTimestamp_timestamp[number3]=timestamp
        number3+=1
    else:
        PV3.auto_monitor=False
        print(pvname+" is finished")
    
PV1=epics.PV(pvname1,auto_monitor=True,callback=GMD1_data_callback)

PV2=epics.PV(pvname2,auto_monitor=True,callback=msoArea_data_callback)

#PV3=epics.PV(pvname3,auto_monitor=True,callback=msoTimestamp_data_callback)

while True:
    if (number1==data_number) and (number2==data_number):
        f.close()
        break
    time.sleep(5)

f=h5py.File(filename,'r')
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
ax5.plot(np.array(f[pvname2+".timestamp"])-np.array(f[pvname1+".timestamp"]),label="timestamp difference")
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
