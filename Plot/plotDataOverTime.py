import numpy as np
import matplotlib.pyplot as plt
def plotDataOverTime(data_increments, data_increment_times, vo_end_time=172):
    opt_traffic = np.cumsum(sum(sum(data_increments[:,:,0])))
    netvlad_traffic = np.cumsum(sum(sum(data_increments[:,:,1])))
    gv_traffic = np.cumsum(sum(sum(data_increments[:,:,2])))
    opt_filter = [True, netvlad_traffic[1:]!=netvlad_traffic[:-1]]
    nv_filt = [True, netvlad_traffic[1:]!= netvlad_traffic[:-1]]
    gv_filt = [True, gv_traffic[1:]!= gv_traffic[:-1]]

    DOpt = plt.plot(data_increment_times[opt_filter], opt_traffic[opt_filter] / 1e6)
    DVPT = plt.plot(data_increment_times[nv_filt], netvlad_traffic[nv_filt] / 1e6)
    RelPose = plt.plot(data_increment_times[gv_filt], gv_traffic[gv_filt] / 1e6)
    [x_min,y_min,x_max,y_max] = plt.axis()
    endTime = plt.plot([vo_end_time, vo_end_time],[y_min,y_max],'k--')
    plt.legend((DOpt,DVPT,RelPose,endTime),('DOpt','DVPT','RelPose','VO end time'))
    plt.title('Total data transmission')
    plt.xlabel('total transmitted [MB]')
