from scipy.ndimage import label
from scipy.ndimage.filters import median_filter,maximum_filter,minimum_filter
from scipy.stats import nanmedian
import pyfits as pf
import numpy
from gaussfitter import onedgaussfit

def noise_calculator(cube,image,in_par):
  image_smooth=median_filter(image,in_par['SMOOTH_KERNEL'])
  mask_zeros=image == 0.#we detect zeros on the image and mask them
  mask_nan= numpy.isnan(image)
  masked0_image=numpy.ma.masked_array(image_smooth,mask_zeros)

  mask_mean= ((image_smooth < nanmedian(masked0_image))*(~mask_zeros))*(~mask_nan)#we calculate the mean of the image ignoring zeros and NaNs
  mask_mean=minimum_filter(mask_mean,in_par['SPREAD_ZONE'])
  
  hdu_back=pf.PrimaryHDU(mask_mean*image)#output of the subtracted sky to check
  hdu_back.writeto('background.fits',clobber=True)
  
  labeled, num_objects = label(~mask_mean)#mask inverted to work with the masked zones instead

  for k in range(1,num_objects+1):
    if len(numpy.where(labeled==k)[0])<in_par['MIN_ZONE']:
      mask_mean[labeled==k]=True#we turn the small zones into true to use them in the noise determination
  
  back_mean=(image[mask_mean]).mean()
  back_noise=(image[mask_mean]).std()
  print 'Noise mean: '+str(back_mean)
  print 'Noise std. var.: '+str(back_noise)

  ##########line noise### ## ## # ## ## ##

  #we find the maxima of the regions
  data_max = maximum_filter(image_smooth,in_par['MAX_KERNEL'])
  maxima=(image_smooth == data_max)
  max_clean=numpy.where(maxima*(data_max>in_par['MAX_MINFLUX']))

  noise_list=[]

  for k in range(len(max_clean[0])):
    i=max_clean[0][k]
    j=max_clean[1][k]
    line=cube[:,i,j]

    x0=numpy.where(line==max(line))[0][0]
    int0=line[x0]

    off=abs(line-0.5*int0)
    d1=numpy.where(off[0:x0]==min(off[0:x0]))
    d2=(numpy.where(off[x0:]==min(off[x0:]))+x0)
    if len(d1)==len(d2)==1:
      disp=0.5*(d2-d1)
      solution=onedgaussfit(range(len(line)),line,params=[0,int0,x0,disp])
      residual=line-solution[1]
      chi=solution[3]

      if residual.mean() < 1e-5 and chi<in_par['CHI_MAX']:
	noise_list.append(residual.std())



  line_noise=numpy.mean(noise_list)
  print 'Line noise std: '+str(line_noise)
  print 'number of points used: '+str(len(noise_list))
  
  return back_mean,back_noise,line_noise
