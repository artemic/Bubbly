import numpy
import matplotlib.pyplot as plt
import deriv
import pyfits as pf
import os
from numpy.ma import median
from numpy import pi
import sys
from scipy.ndimage.filters import gaussian_filter
sys.path.append('mpfit.py')#necesario? deriv no lo usa
from mpfit import mpfit
from gaussfitter import onedgaussfit,n_gaussian,multigaussfit
from functions import *

#Read input parameters from file
in_par_list=open('parfile.conf').readlines()
in_par_list=in_par_list[2:]
in_par={}
for line in in_par_list:
  if not line.startswith('#'):
    key=line[0:line.find('=')].strip()
    value=line[line.find('=')+1:line.find('#')].strip()
    in_par[key]=param_parser(value)


#Read data cube and intensity map, if any
cube_fits=pf.open(in_par['CUBE_NAME'])[0]
cube=cube_fits.data
cube_hdr=cube_fits.header
cube=numpy.squeeze(cube)#eliminate 1d entries (48,1024,1024,1)==> (48,1024,1024)
#create the header for the output fits
hdr_map=header_create(cube_hdr,in_par_list)

if not in_par['RERUN']:
  if in_par['MONO_MAP'] == 1:
    mono=pf.open(in_par['MONO_NAME'])[0].data
  else:
    mono=numpy.sum(cube,0)


#smooth cube if necessary
if in_par['SMOOTH']:
  kernel=[0,0,0]
  kernel[1]=kernel[2]=in_par['XY_KERNEL']
  kernel[0]=in_par['SPEC_KERNEL']
  cube=gaussian_filter(cube,kernel)
  mono=gaussian_filter(mono,kernel[1:])


#Produce mask and obtain spectral noise parameters
if not in_par['RERUN']:
  if in_par['NOISE_DET']==0:
    cutoff=in_par['MINFLUX']
    SNR_S=float(in_par['SNR_S'])
    spec_noise=float(in_par['NOISE_S'])
    mask=mono>cutoff
  elif in_par['NOISE_DET']==1:
    print 'separating galaxy emission and calculating spectral noise'
    SNR_S=float(in_par['SNR_S'])
    cube,mask,spec_noise=noise_calculator(cube,hdr_map,in_par)

sizez,sizex,sizey=cube.shape

#Determine the vector of velocities corresponding to the spectra
vel,v_res=vel_profile(cube_hdr,sizez,in_par)

#proceed to multi-peak detection and fitting unless the code is rerun on the same multi maps:
if not in_par['RERUN']:
  print 'performing peak detectiong and fitting'
  #Initialize dictionaries and lists where we will store data for the calculations
  zeromap=numpy.zeros([sizex,sizey])
  vel_map=numpy.zeros([1,sizex,sizey])
  disp_map=numpy.array(vel_map)
  mono_map=numpy.array(vel_map)

  limitedmin=[True,True,True]
  limitedmax=[True,True,True]
  
  for i in range(sizex):
    for j in range(sizey):
      if mask[i,j]:
	perf=cube[:,i,j]
	if in_par['CENTERING'] == 1:
	  vel_p,perf_p=line_centering(vel,perf)
	else:
	  vel_p,perf_p=vel,perf
	profile_i=profile(vel_p,perf_p)
	num_zeros=profile_i.get_zeros(SNR_S*spec_noise,10)
	if num_zeros > 1:
	  params,minpars,maxpars=get_pars(profile_i,num_zeros,v_res)
	  ind_max=numpy.argmax(params[numpy.arange(0,len(params),3)])#we impose a higher lower limit to the main peak to ensure it is the most intense after fitting
	  minpars[3*ind_max]=0.9*params[3*ind_max]
	  limitedmin=[True,True,True]
	  limitedmax=[True,True,True]
	  #we perform the first fit, which we will use to subtract the main peak and then perform the peak detection again
	  fit_params=profile_i.gaussianfit(num_zeros,list(params),limitedmin,limitedmax,list(minpars),list(maxpars))
	  #we redo the peak detection subtracting the main peak
	  perf_2=perf_p-fit_params[3*ind_max]*numpy.exp(-((vel_p-fit_params[3*ind_max+1])**2.)/(2.*fit_params[3*ind_max+2]**2.))
	  profile_i=profile(vel_p,perf_2)
	  num_zeros2=profile_i.get_zeros(SNR_S*spec_noise,10)
	  if num_zeros2 > 1:
	    params,minpars,maxpars=get_pars(profile_i,num_zeros2,v_res)
	    params,minpars,maxpars,num_zeros=fix_primpars(params,minpars,maxpars,fit_params,num_zeros2,ind_max,v_res)

	    profile_i=profile(vel_p,perf_p)
	    limitedmin=[True,True,True]
	    limitedmax=[True,True,True]
	    #second and final fit with updated parameters
	    fit_params=profile_i.gaussianfit(num_zeros,list(params),limitedmin,limitedmax,list(minpars),list(maxpars))
	    
	  if num_zeros/2>vel_map.shape[0]:
	    for rep in range(num_zeros/2-vel_map.shape[0]):
	      vel_map=append_map(vel_map,zeromap)
	      disp_map=append_map(disp_map,zeromap)
	      mono_map=append_map(mono_map,zeromap)
	    
	  for zeros in range(num_zeros/2):
	    vel_map[zeros,i,j]=fit_params[zeros*3+1]
	    disp_map[zeros,i,j]=fit_params[zeros*3+2]
	    mono_map[zeros,i,j]=fit_params[zeros*3]*fit_params[zeros*3+2]


  map_number=vel_map.shape[0]
  #We calibrate in intensity, since mono is already int*disp, the calibration is (a*I+b)*c*D=a*c*I*D+b*c*D=a*c*MONO+b*c*D
  mono_map=(pi**0.5)*(mono_map*in_par['CALIB_A']*in_par['CALIB_C']+in_par['CALIB_B']*in_par['CALIB_C']*disp_map)
  cube=cube*in_par['CALIB_A']+in_par['CALIB_B']
  if in_par['COMP_OUT'] == 1:
        
    try:
      os.mkdir('multiple_maps')
    except:
      os.system('rm multiple_maps/*')
    for component in range(map_number):
      pf.writeto('multiple_maps/vel'+str(component)+'.fits',vel_map[component,:,:],hdr_map,clobber=True)
      pf.writeto('multiple_maps/disp'+str(component)+'.fits',disp_map[component,:,:],hdr_map,clobber=True)
      pf.writeto('multiple_maps/mono'+str(component)+'.fits',mono_map[component,:,:],hdr_map,clobber=True)


#BUBBLES

#recover component maps if the program is re-run on an existing set of maps
if in_par['RERUN']:
  print 'reading multi-peak maps'
  map_number=len(os.listdir('multiple_maps'))/3
  vel_map=numpy.zeros([map_number,sizex,sizey])
  mono_map=numpy.zeros([map_number,sizex,sizey])
  disp_map=numpy.zeros([map_number,sizex,sizey])
  for k in numpy.arange(0,map_number):
    fitk_v=pf.open('multiple_maps/vel'+str(k)+'.fits')
    vel_map[k,:,:]=fitk_v[0].data
    fitk_i=pf.open('multiple_maps/mono'+str(k)+'.fits')
    mono_map[k,:,:]=fitk_i[0].data
    fitk_d=pf.open('multiple_maps/disp'+str(k)+'.fits')
    disp_map[k,:,:]=fitk_d[0].data
  
  hdr_prev=fitk_v[0].header
  
  hdr_map=header_correct(hdr_prev,hdr_map)#make correct header from parameters for the multiple maps and the current parfile


print 'maximum number of peaks: ',map_number


#We start to construct the main peak maps, we begin by assigning the values of where there is only one peak detected
main_v=numpy.array(vel_map[0,:,:])
main_i=numpy.array(mono_map[0,:,:])
main_d=numpy.array(disp_map[0,:,:])

if in_par['MAIN_EXT'] == 0 or in_par['MAIN_OUT'] == 1:
  for i in range(sizex):#We assign the brightest peak in each pixel to the main peak map
    for j in range(sizey):
      if (sum(mono_map[1:,i,j]) != 0): 
	perf=cube[:,i,j]
	v_imax=vel[numpy.where(perf==max(perf))]
	vdiff=abs(vel_map[:,i,j]-v_imax)
	min_vdiff=min(vdiff[vdiff!=abs(v_imax)])
	nmap=numpy.where((vdiff==min_vdiff))[0][0]
	
	main_v[i,j]=vel_map[nmap,i,j]
	main_i[i,j]=mono_map[nmap,i,j]
	main_d[i,j]=disp_map[nmap,i,j]
	#vel_map[nmap,i,j]=0.
	#mono_map[nmap,i,j]=0.
	#disp_map[nmap,i,j]=0.
  print 'main peak maps produced'
      
if in_par['MAIN_OUT'] == 1:
  mask0=numpy.where(main_v==0.)
  main_v[mask0]=numpy.nan
  main_i[mask0]=numpy.nan
  main_d[mask0]=numpy.nan
  pf.writeto('main_v.fits',main_v,hdr_map,clobber=True)
  pf.writeto('main_d.fits',main_d,hdr_map,clobber=True)
  pf.writeto('main_i.fits',main_i,hdr_map,clobber=True)
  main_v[mask0]=0.
  main_i[mask0]=0.
  main_d[mask0]=0.


if in_par['MAIN_EXT'] == 1:
  main_v=pf.open(in_par['MAIN_NAME'])[0].data

print 'detecting expansion'
veloff=numpy.array([numpy.subtract(vel_map[k,:,:],main_v) for k in numpy.arange(map_number)])
veloff_abs=abs(veloff)
bub=[]
for m1 in range(map_number-1):#we look for matches to the conditions for expansion presence
  for m2 in range(m1+1,map_number):
    bub_m=numpy.where((abs(veloff_abs[m1,:,:]-veloff_abs[m2,:,:]) < in_par['BUB_V_SIGMA']*v_res) & (veloff_abs[m1,:,:]-veloff_abs[m2,:,:] != 0.) & (veloff_abs[m1,:,:]!=0.) & (veloff_abs[m2,:,:]!=0.) & ((veloff[m1,:,:]<0) !=(veloff[m2,:,:]<0)) & (veloff_abs[m1,:,:]!=abs(main_v)) & (veloff_abs[m2,:,:]!=abs(main_v)))
    bub_m=[(bub_m[0][i1],bub_m[1][i1],m1,m2) for i1 in range(len(bub_m[0]))]
    bub.extend(bub_m)

if in_par['BUB_USE_I'] == 1:
  bub_i=[]
  for m1 in range(map_number-1):
    for m2 in range(m1+1,map_number):
      bub_m_i=numpy.where((abs(mono_map[m1,:,:]-mono_map[m2,:,:]))/(mono_map[m1,:,:]+mono_map[m2,:,:]) < in_par['BUB_I_R'])
      bub_m_i=[(bub_m_i[0][i1],bub_m_i[1][i1],m1,m2) for i1 in range(len(bub_m_i[0]))]
      bub_i.extend(bub_m_i)
  bub=list(set(bub)&set(bub_i))


if in_par['BUB_USE_D'] == 1:
  bub_d=[]
  for m1 in range(map_number-1):
    for m2 in range(m1+1,map_number):
      bub_m_d=numpy.where((abs(disp_map[m1,:,:]-disp_map[m2,:,:]))/(disp_map[m1,:,:]+disp_map[m2,:,:]) < in_par['BUB_D_R'])
      bub_m_d=[(bub_m_d[0][i1],bub_m_d[1][i1],m1,m2) for i1 in range(len(bub_m_d[0]))]
      bub_d.extend(bub_m_d)
  bub=list(set(bub)&set(bub_d))

#we create the arrays that will contain the information about the detected expansions
nullmap=numpy.empty([sizex,sizey])
nullmap.fill(numpy.nan)
bubble_map=numpy.array(nullmap)
bubble_intrel_map=numpy.array(nullmap)
bubble_disp_map=numpy.array(nullmap)
bubble_sec=numpy.empty([1,sizex,sizey])
bubble_sec.fill(numpy.nan)
bubble_sec_int=numpy.array(bubble_sec)
bubble_sec_disp=numpy.array(bubble_sec)

for coord in bub:
  i=coord[0]
  j=coord[1]
  m1=coord[2]
  m2=coord[3]
  voff=0.5*veloff_abs[m1,i,j]+0.5*veloff_abs[m2,i,j]
  b_disp=0.5*disp_map[m1,i,j]+0.5*disp_map[m2,i,j]
  if in_par['INT_TYPE']==0:
    irel=(mono_map[m1+1,i,j]+mono_map[m2+1,i,j])/(2.*main_i[i,j])
  elif in_par['INT_TYPE']==1:
    irel=(mono_map[m1+1,i,j]+mono_map[m2+1,i,j])/2.
    irel=(pi**0.5)*(irel*in_par['CALIB_A']*in_par['CALIB_C']+in_par['CALIB_B']*in_par['CALIB_C']*disp_map[m2+1,i,j])
    
  if numpy.isnan(bubble_map[i,j]):#we check whether there is already a detection at that point and add additional maps if necessary
    bubble_map[i,j]=voff
    bubble_intrel_map[i,j]=irel
    bubble_disp_map[i,j]=b_disp
  else:
    if not numpy.isnan(sum(bubble_sec[:,i,j])):
      bubble_sec=append_map(bubble_sec,nullmap)
      bubble_sec_int=append_map(bubble_sec_int,nullmap)
      bubble_sec_disp=append_map(bubble_sec_disp,nullmap)
      
    if voff<bubble_map[i,j]:#we select the expansion with the highest velocity to be on the main map
      bubsec_index=min(numpy.where(numpy.isnan(bubble_sec[:,i,j]))[0])
      bubble_sec[bubsec_index,i,j]=voff
      bubble_sec_int[bubsec_index,i,j]=irel
      bubble_sec_disp[bubsec_index,i,j]=b_disp
    elif voff==bubble_map[i,j]:
      print coord
    else:
      bubsec_index=min(numpy.where(numpy.isnan(bubble_sec[:,i,j]))[0])
      bubble_sec[bubsec_index,i,j]=bubble_map[i,j]
      bubble_sec_int[bubsec_index,i,j]=bubble_intrel_map[i,j]
      bubble_sec_disp[bubsec_index,i,j]=bubble_disp_map[i,j]
      bubble_map[i,j]=voff
      bubble_intrel_map[i,j]=irel
      bubble_disp_map[i,j]=b_disp


#bubble maps are complete
pf.writeto('bubble.fits',bubble_map,hdr_map,clobber=True)
pf.writeto('bubble_int.fits',bubble_intrel_map,hdr_map,clobber=True)
pf.writeto('bubble_disp.fits',bubble_disp_map,hdr_map,clobber=True)

hdr_map['NAXIS']=3
hdr_map['NAXIS3']=bubble_sec.shape[0]

pf.writeto('bubble_sec.fits',bubble_sec,hdr_map,clobber=True)
pf.writeto('bubble_sec_int.fits',bubble_sec_int,hdr_map,clobber=True)
pf.writeto('bubble_sec_disp.fits',bubble_sec_int,hdr_map,clobber=True)




if in_par['PROF_OUT'] == 1:
  print 'outputting fitted profiles with expansion'
  try:
    os.mkdir('profiles')
  except:
    os.system('rm profiles/*')
  for coord in bub:
    i=coord[0]
    j=coord[1]
    plot_profile(cube,vel,i,j,in_par['CENTERING'],vel_map,mono_map,disp_map,in_par)
