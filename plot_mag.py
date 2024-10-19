def mag_cal_plot():
    plt.style.use('ggplot') # start figure
    fig,axs = plt.subplots(1,2,figsize=(12,7)) # start figure
    for mag_ii,mags in enumerate(mag_cal_rotation_vec):
        mags = np.array(mags) # magnetometer numpy array
        x,y = mags[:,cal_rot_indices[mag_ii][0]],\
                    mags[:,cal_rot_indices[mag_ii][1]]
        x,y = outlier_removal(x,y) # outlier removal 
        axs[0].scatter(x,y,
                       label='Rotation Around ${0}$-axis (${1},{2}$)'.\
                    format(mag_cal_axes[mag_ii],
                           mag_labels[cal_rot_indices[mag_ii][0]],
                           mag_labels[cal_rot_indices[mag_ii][1]]))
        axs[1].scatter(x-mag_coeffs[cal_rot_indices[mag_ii][0]],
                    y-mag_coeffs[cal_rot_indices[mag_ii][1]],
                       label='Rotation Around ${0}$-axis (${1},{2}$)'.\
                    format(mag_cal_axes[mag_ii],
                           mag_labels[cal_rot_indices[mag_ii][0]],
                           mag_labels[cal_rot_indices[mag_ii][1]]))
    axs[0].set_title('Before Hard Iron Offset') # plot title
    axs[1].set_title('After Hard Iron Offset') # plot title
    mag_lims = [np.nanmin(np.nanmin(mag_cal_rotation_vec)),
                np.nanmax(np.nanmax(mag_cal_rotation_vec))] # array limits
    mag_lims = [-1.1*np.max(mag_lims),1.1*np.max(mag_lims)] # axes limits
    for jj in range(0,2):
        axs[jj].set_ylim(mag_lims) # set limits
        axs[jj].set_xlim(mag_lims) # set limits
        axs[jj].legend() # legend
        axs[jj].set_aspect('equal',adjustable='box') # square axes
    fig.savefig('mag_cal_hard_offset.png',dpi=300,bbox_inches='tight',
                facecolor='#FCFCFC') # save figure
    fig.savefig('mag_cal_hard_offset_white.png',dpi=300,bbox_inches='tight',
                facecolor='#FFFFFF') # save figure
    plt.show() #show plot   