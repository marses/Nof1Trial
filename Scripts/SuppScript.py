import numpy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
from datetime import datetime


def inverse_quaternion(quaternion):
    # inverse of a quaternion
    
    quaternion = quaternion.reshape(-1,4)
    out = numpy.empty((0, 4))
    for i in range(0,len(quaternion)):
        r = 1/(quaternion[i,0]**2+quaternion[i,1]**2+quaternion[i,2]**2+quaternion[i,3]**2)*numpy.array([-quaternion[i,0],-quaternion[i,1],-quaternion[i,2],quaternion[i,3]])
        out = numpy.vstack([out,r])
    return out


def quaternion_multiply(quaternion1, quaternion2):
    # multiplication of two quaternions
    
    quaternion1 = quaternion1.reshape(-1,4)
    quaternion2 = quaternion2.reshape(-1,4)
    
    out = numpy.empty((0, 4))
    for i in range(0,len(quaternion1)):
        for j in range(0,len(quaternion2)):
            b1,c1,d1,a1 = quaternion1[i,:]
            b2,c2,d2,a2 = quaternion2[j,:]
            r = numpy.array([a1*b2+b1*a2+c1*d2-d1*c2,
        			a1*c2-b1*d2+c1*a2+d1*b2,
        			a1*d2+b1*c2-c1*b2+d1*a2,
        			a1*a2-b1*b2-c1*c2-d1*d2])
            out = numpy.vstack([out,r])
    return out

def quaternion_to_euler(quaternion):
    # convert quaternion to Euler angles
    
    quaternion = quaternion.reshape(-1,4)
    import math
    out = numpy.empty((0, 3))
    for i in range(0,len(quaternion)):
        x, y, z, w = quaternion[i,:]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = (math.atan2(t0, t1))
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = (math.asin(t2))
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = (math.atan2(t3, t4))
        r = numpy.array([X, Y, Z])
        out = numpy.vstack([out,r])
    return out



def euler_to_quaternion(angle):
    # convert Euler angle(s) to quaternion
    
    out = numpy.empty((0, 4))
    for i in range(0,len(angle)):
        roll, pitch, yaw = angle[i,:]
        qx = numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) - numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
        qy = numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2)
        qz = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2) - numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2)
        qw = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
        r = numpy.array([qx, qy, qz, qw])
        out = numpy.vstack([out,r])
    return out

###


def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).

    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.

    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.

    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.

    See this IPython Notebook [2]_.

    References
    ----------
    .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

    """
    
    x = numpy.atleast_1d(x).astype('float64')
    gp, gn = numpy.zeros(x.size), numpy.zeros(x.size)
    ta, tai, taf = numpy.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = numpy.array([])
    a = -0.0001
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < a:
            gp[i], tap = 0, i
        if gn[i] < a:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = numpy.append(ta, i)    # alarm index
            tai = numpy.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = numpy.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = numpy.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[numpy.argmax(taf >= i) for i in ta]]
            else:
                ind = [numpy.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~numpy.append(False, ind)]
            tai = tai[~numpy.append(False, ind)]
            taf = taf[~numpy.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    if show:
        _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

    return ta, tai, taf, amp


def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
    """Plot results of the detect_cusum function, see its help."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                         label='Ending')
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[numpy.isfinite(x)].min(), x[numpy.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax2.set_xlim(-.01*x.size, x.size*1.01-1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()

##



def peaks(y,t):
    """
    Finds peaks in the signal.
    """

    finite_difference_1 = numpy.gradient(y, t)
    
    is_peak = [finite_difference_1[i] * finite_difference_1[i + 1] <= -0*0.0001 for i in range(len(finite_difference_1) - 1)]

    peak_indices = [i for i, b in enumerate(is_peak) if b]

    if len(peak_indices) == 0:
        return [],[]
        
    return peak_indices, y[peak_indices]




def inflection_points(y, t):
    '''
    Find the list of vertices that preceed inflection points in a curve. 
    
    returns: a list of points in space corresponding to the vertices that
    immediately preceed inflection points in the curve
    '''
    # Take the second order finite difference of the curve with respect to the
    # defined coordinate system

    finite_difference_2 = numpy.gradient(numpy.gradient(y, t), t)

    # Compare the product of all neighboring pairs of points in the second derivative
    # If a pair of points has a negative product, then the second derivative changes sign
    # at one of those points, signalling an inflection point
    is_inflection_point = [finite_difference_2[i] * finite_difference_2[i + 1] <= -0.001 for i in range(len(finite_difference_2) - 1)]

    inflection_point_indices = [i for i, b in enumerate(is_inflection_point) if b]

    if len(inflection_point_indices) == 0: # pylint: disable=len-as-condition
        return []

    return numpy.array(inflection_point_indices), y[inflection_point_indices]




def onset_peaks(y,t):
	"""
		y and t are numpy arrays
	"""
	# if the window is shorter than 10 samples, return the final index
	if len(y) < 15:
		return len(y)-1

	if len(y) < 15:
		l_filter = int(numpy.ceil(len(y)) // 2 * 2 - 1)
	else:
		l_filter = 15

	turning_points = peaks(y,t)	
	if len(turning_points[0]) > 0:
		peak_index, _ = turning_points
		if peak_index[-1] >= 18:
			return peak_index[-1]
		else:
			y_filtered = savgol_filter(y, l_filter, 3)
			turning_points_filtered = peaks(y_filtered,t)
			if len(turning_points_filtered[0]) >0:
				peak_index_filtered, _ = turning_points_filtered
				if peak_index_filtered[-1] >= 18:
					return peak_index_filtered[-1]
				else:
					# normalize the signal y
					y_normalized = (y-y.min())/(y.max()-y.min())
						# call CumSum
					ta, tai, taf, amp = detect_cusum(y_normalized[10:], 0.045, .000175, True, False)
					if ta[-1] > 5:
						return ta[-1]+5
					else:
						return len(y)-1

			else:
					# normalize the signal y
				y_normalized = (y-y.min())/(y.max()-y.min())
					# call CumSum
				ta, tai, taf, amp = detect_cusum(y_normalized, 0.045, .000175, True, False)
				if ta[-1] > 15:
					return ta[-1]
				else:
					return len(y)-1
	else:
		y_filtered = savgol_filter(y, l_filter, 3)
		turning_points_filtered = peaks(y_filtered,t)
		if len(turning_points_filtered[0]) > 0:
			peak_index_filtered, _ = turning_points_filtered
			if peak_index_filtered[-1] >= 18:
				return peak_index_filtered[-1]
			else:
				y_normalized = (y-y.min())/(y.max()-y.min())
				# call CumSum
				ta, tai, taf, amp = detect_cusum(y_normalized[10:], 0.045, .000175, True, False)
				if ta[-1] > 5:
					return ta[-1] + 5
				else:
					return len(y)-1
		else:
			y_normalized = (y-y.min())/(y.max()-y.min())
			# call CumSum
			ta, tai, taf, amp = detect_cusum(y_normalized[10:], 0.045, .000175, True, False)
			if ta[-1] > 5:
				return ta[-1]+10
			else:
				return len(y)-1



def number_of_flips(y,t):
    """
    Finds number of flips in a signal.
    """

    finite_difference_1 = numpy.gradient(y, t)
    
    is_peak = [finite_difference_1[i] * finite_difference_1[i + 1] <= -0*0.0001 for i in range(len(finite_difference_1) - 1)]

    peak_indices = [i for i, b in enumerate(is_peak) if b]

    if len(peak_indices) == 0:
        return 0
        
    return len(peak_indices)
	