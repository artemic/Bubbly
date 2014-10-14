import numpy
import matplotlib.pyplot as plt
import deriv
import pyfits as pf
import os
from numpy.ma import median
from numpy import pi
import sys
from scipy.ndimage.filters import gaussian_filter
sys.path.append('mpfit.py')
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

z_index=in_par['Z_IND']
#Read data cube and intensity map, if any
cube_fits=pf.open(in_par['CUBE_NAME'])[0]
cube=cube_fits.data
cube_hdr=cube_fits.header
#reorganize indexes to agree with the expected dimensions
cube=numpy.squeeze(cube)#eliminate 1d entries (48,1024,1024,1)==> (48,1024,1024)
cube=numpy.rollaxis(cube,z_index-1,0)#put z-index in position 0 without modifying the array
#smooth cube if necessary

if in_par['MONO_MAP'] == 1:
  mono=pf.open(in_par['MONO_NAME'])[0].data
else:
  mono=numpy.sum(cube,int(in_par['Z_IND'])-1)

if in_par['SMOOTH']:
  kernel=[0,0,0]
  kernel[1]=kernel[2]=in_par['XY_KERNEL']
  kernel[0]=in_par['SPEC_KERNEL']
  cube=gaussian_filter(cube,kernel)
  mono=gaussian_filter(mono,kernel[1:])

#Read different noise parameters
if in_par['NOISE_DET']==0:
  SNR=float(in_par['SNR'])
  back_mean=float(in_par['BACK_MEAN'])
  back_noise=float(in_par['BACK_NOISE'])
  cutoff=back_mean+SNR*back_noise
  SNR_S=float(in_par['SNR_S'])
  spec_noise=float(in_par['NOISE_S'])
elif in_par['NOISE_DET']==1:
  from noise import *
  SNR=float(in_par['SNR'])
  SNR_S=float(in_par['SNR_S'])
  back_mean,back_noise,spec_noise=noise_calculator(cube,mono,in_par)
  cutoff=back_mean+SNR*back_noise

sizex=cube.shape[1]
sizey=cube.shape[2]
sizez=cube.shape[0]

#create the header for the output fits
hdr_map=header_create(cube_hdr,in_par_list)

#indexz=numpy.arange(0.,float(sizez),1)

#Determine the vector of velocities corresponding to the spectra
vel,v_res=vel_profile(cube_hdr,z_index,sizez,in_par)

#Initialize dictionaries and lists where we will store data for the calculations
zero_dict={}
vel_dict={}
disp_dict={}
peak_dict={}
vel_dict_gauss={}
disp_dict_gauss={}
peak_dict_gauss={}
position_list=[]

#proceed to multi-peak detection and fitting unless the code is rerun on the same multi maps:
if not in_par['RERUN']:

  for i in range(sizex):
    for j in range(sizey):
      if mono[i,j] >=cutoff:
	perf=cube[:,i,j]
	if in_par['CENTERING'] == 1:
	  vel_p,perf_p=line_centering(vel,perf)
	profile_i=profile(vel_p,perf_p)
	num_zeros=profile_i.get_zeros(SNR_S*spec_noise,10)
	zero_dict[(i,j)]=num_zeros
	for zeros in range(num_zeros/2):
	  if zeros == 0:
	    vel_dict[(i,j)]=(0.5*(profile_i.zero[2*zeros]+profile_i.zero[2*zeros+1]))
	    disp_dict[(i,j)]=(0.5*(profile_i.zero[2*zeros+1]-profile_i.zero[2*zeros]))
	    peak_dict[(i,j)]=profile_i.peak[zeros]
	  elif zeros==1:
	    vel_dict[(i,j)]=(vel_dict[(i,j)],0.5*(profile_i.zero[2*zeros]+profile_i.zero[2*zeros+1]))
	    disp_dict[(i,j)]=(disp_dict[(i,j)],0.5*(profile_i.zero[2*zeros+1]-profile_i.zero[2*zeros]))
	    peak_dict[(i,j)]=(peak_dict[(i,j)],profile_i.peak[zeros])
	  else:
	    temporal_list=list(vel_dict[(i,j)])
	    temporal_list.append(0.5*(profile_i.zero[2*zeros]+profile_i.zero[2*zeros+1]))
	    vel_dict[(i,j)]=tuple(temporal_list)
	    temporal_list=list(disp_dict[(i,j)])
	    temporal_list.append(0.5*(profile_i.zero[2*zeros+1]-profile_i.zero[2*zeros]))
	    disp_dict[(i,j)]=tuple(temporal_list)
	    temporal_list=list(peak_dict[(i,j)])
	    temporal_list.append(profile_i.peak[zeros])
	    peak_dict[(i,j)]=tuple(temporal_list)
	    
  # En este momento tengo los valores iniciales de los parametros y los limites. Calculo los limites y ajusto gaussianas multilples
	params=numpy.arange(float(int(3*(num_zeros/2))))
	minpars=numpy.arange(float(int(3*(num_zeros/2))))
	maxpars=numpy.arange(float(int(3*(num_zeros/2))))
	limitedmin=[True,True,True]
	limitedmax=[True,True,True]
	
	for index in range(num_zeros/2):
	  if num_zeros > 3:
	    params[3*index+1]=vel_dict[(i,j)][index]
	    params[3*index]=peak_dict[(i,j)][index]
	    if disp_dict[(i,j)][index] >= v_res:
	      params[3*index+2]=disp_dict[(i,j)][index]
	    else:
	      params[3*index+2]=v_res
	    
	    minpars[3*index]=0.0#el valor minimo del pico lo poco a cero
	    minpars[3*index+1]=profile_i.zero[2*index] #el valor minimo de la velocidad media lo pongo en el primer cero de la segunda derivada
	    if disp_dict[(i,j)][index] >= v_res:
	      minpars[3*index+2]=0.5*disp_dict[(i,j)][index] #el valor minimo de la anchura lo pongo a la mitad de la estimada segun los ceros de 
	    else:#la segunda derivada
	      minpars[3*index+2]=v_res/2.

	    maxpars[3*index]=1.5*peak_dict[(i,j)][index]#el valor maximo de la amplitud lo pongo como 1.5*el valor estimado con ceros                                              
	    maxpars[3*index+1]=profile_i.zero[2*index+1]#el valor maximo de la velocidad media lo pongo en el segundo cero de la segunda derivada
	    if disp_dict[(i,j)][index] >= v_res:
	      maxpars[3*index+2]=2.*disp_dict[(i,j)][index]#el valor maximo de la anchura lo pongo como el doble de la estimada segun los ceros de 
	    else:#la segunda derivada
	      maxpars[3*index+2]=2*v_res

	  else:
	    params[3*index+1]=vel_dict[(i,j)]
	    params[3*index]=peak_dict[(i,j)]
	    if disp_dict[(i,j)] >= v_res:
	      params[3*index+2]=disp_dict[(i,j)]#el valor maximo de la anchura lo pongo como el doble de la estimada segun los ceros de 
	    else:#params[3*index+2]=disp_dict[(i,j)]
	      params[3*index+2]=v_res
	    
	    minpars[3*index]=0.0#el valor minimo del pico lo poco a cero
	    minpars[3*index+1]=profile_i.zero[2*index] #el valor minimo de la velocidad media lo pongo en el primer cero de la segunda derivada
	    if disp_dict[(i,j)] >= v_res:
	      minpars[3*index+2]=0.5*disp_dict[(i,j)] #el valor minimo de la anchura lo pongo a la mitad de la estimada segun los ceros de la segunda derivada
	    else:
	      minpars[3*index+2]=v_res/2.
	    
	    maxpars[3*index]=1.5*peak_dict[(i,j)]#el valor maximo de la amplitud lo pongo como 1.5*el valor estimado con ceros                                              
	    maxpars[3*index+1]=profile_i.zero[2*index+1]#el valor maximo de la velocidad media lo pongo en el segundo cero de la segunda derivada
	    if disp_dict[(i,j)] >= v_res:
	      maxpars[3*index+2]=2.*disp_dict[(i,j)]#el valor maximo de la anchura lo pongo como el doble de la estimada segun los ceros de 
	    else:
	      maxpars[3*index+2]=2*v_res
	    
	    
	if num_zeros > 1:
	  ints=[params[index] for index in numpy.arange(0,len(params),3)]
	  ind_max=numpy.where(numpy.array(ints)==max(numpy.array(ints)))[0]#imponemos un limite inferior al pico principal para que sea el de mayor intensidad
	  minpars[3*ind_max]=0.9*max(numpy.array(ints))
	  position_list.append((i,j))
	  fit_params=profile_i.gaussianfit(num_zeros,list(params),limitedmin,limitedmax,list(minpars),list(maxpars))
	for zeros in range(num_zeros/2):
	  if zeros == 0:
	    vel_dict_gauss[(i,j)]=fit_params[zeros*3+1]
	    disp_dict_gauss[(i,j)]=fit_params[zeros*3+2]
	    peak_dict_gauss[(i,j)]=fit_params[zeros*3]
	  elif zeros==1:
	    vel_dict_gauss[(i,j)]=(vel_dict_gauss[(i,j)],fit_params[zeros*3+1])
	    disp_dict_gauss[(i,j)]=(disp_dict_gauss[(i,j)],fit_params[zeros*3+2])
	    peak_dict_gauss[(i,j)]=(peak_dict_gauss[(i,j)],fit_params[zeros*3])
	  else:
	    temporal_list=list(vel_dict_gauss[(i,j)])
	    temporal_list.append(fit_params[zeros*3+1])
	    vel_dict_gauss[(i,j)]=tuple(temporal_list)
	    temporal_list=list(disp_dict_gauss[(i,j)])
	    temporal_list.append(fit_params[zeros*3+2])
	    disp_dict_gauss[(i,j)]=tuple(temporal_list)
	    temporal_list=list(peak_dict_gauss[(i,j)])
	    temporal_list.append(fit_params[zeros*3])
	    peak_dict_gauss[(i,j)]=tuple(temporal_list)
  #
  #The fit is complete for all points
  #We produce the maps

  max_zeros=max(zero_dict.values())
  map_number=sum(numpy.arange(1,max_zeros/2+1))
  #print 'The number of maps  is ', map_number 
  vel_map=numpy.zeros((map_number,sizex,sizey))
  disp_map=numpy.zeros((map_number,sizex,sizey))
  mono_map=numpy.zeros((map_number,sizex,sizey))
  for position in position_list:
    if zero_dict[position] > 3:
      for component in range(zero_dict[position]/2):

	vel_map[sum(range(zero_dict[position]/2))+component,position[0],position[1]]=vel_dict_gauss[position][component]
	disp_map[sum(range(zero_dict[position]/2))+component,position[0],position[1]]=disp_dict_gauss[position][component]
	mono_map[sum(range(zero_dict[position]/2))+component,position[0],position[1]]=(disp_dict_gauss[position][component])*(peak_dict_gauss[position][component])
    else:
      vel_map[0,position[0],position[1]]=vel_dict_gauss[position]
      disp_map[0,position[0],position[1]]=disp_dict_gauss[position]
      mono_map[0,position[0],position[1]]=(disp_dict_gauss[position])*(peak_dict_gauss[position])

  if in_par['COMP_OUT'] == 1:
    #Since mono is already int*disp, the calibration is (a*I+b)*c*D=a*c*I*D+b*c*D=a*c*MONO+b*c*D
    mono_out=(pi**0.5)*(mono_map*in_par['CALIB_A']*in_par['CALIB_C']+in_par['CALIB_B']*in_par['CALIB_C']*disp_map)
    
    try:
      os.mkdir('multiple_maps')
    except:
      os.system('rm multiple_maps/*')
    for component in range(map_number):
      pf.writeto('multiple_maps/vel'+str(component)+'.fits',vel_map[component,:,:],hdr_map,clobber=True)
      pf.writeto('multiple_maps/disp'+str(component)+'.fits',disp_map[component,:,:],hdr_map,clobber=True)
      pf.writeto('multiple_maps/mono'+str(component)+'.fits',mono_out[component,:,:],hdr_map,clobber=True)
      #hdu_vel=pf.PrimaryHDU(vel_map[component,:,:])
      #hdu_disp=pf.PrimaryHDU(disp_map[component,:,:])
      #hdu_mono=pf.PrimaryHDU(mono_map[component,:,:])
      #hdu_vel.writeto('multiple_maps/vel'+str(component)+'.fits',hdr_map,clobber=True)
      #hdu_disp.writeto('multiple_maps/disp'+str(component)+'.fits',hdr_map,clobber=True)
      #hdu_mono.writeto('multiple_maps/mono'+str(component)+'.fits',hdr_map,clobber=True)


#BUBBLES

#recover component maps if the program is re-run on an existing set of maps
if in_par['RERUN']:
  map_number=len(os.listdir('multiple_maps'))/3
  vel_map=numpy.zeros([map_number,sizex,sizey])
  mono_map=numpy.zeros([map_number,sizex,sizey])
  disp_map=numpy.zeros([map_number,sizex,sizey])
  for k in numpy.arange(0,map_number):
    #fitk_v=pf.open('multiple_results/vel'+str(k)+'.fits')
    #vel_map[:,:,k]=fitk_v[0].data
    #fitk_i=pf.open('multiple_results/mono'+str(k)+'.fits')
    #mono_map[:,:,k]=fitk_i[0].data
    #fitk_d=pf.open('multiple_results/disp'+str(k)+'.fits')
    #disp_map[:,:,k]=fitk_d[0].data
    fitk_v=pf.open('multiple_maps/vel'+str(k)+'.fits')
    vel_map[k,:,:]=fitk_v[0].data
    fitk_i=pf.open('multiple_maps/mono'+str(k)+'.fits')
    mono_map[k,:,:]=fitk_i[0].data
    fitk_d=pf.open('multiple_maps/disp'+str(k)+'.fits')
    disp_map[k,:,:]=fitk_d[0].data
  
  hdr_prev=fitk_v[0].header
  
  hdr_map=header_correct(hdr_prev,hdr_map)#make correct header from parameters for the multiple maps and the current parfile





print 'maximum number of peaks: ',num_peaks(map_number,0,1)

#arreglar formatos de mapas!!!

#We start to construct the main peak maps, we begin by assigning the values of where there is only one peak detected
main_v=vel_map[0,:,:]
main_i=mono_map[0,:,:]
main_d=disp_map[0,:,:]

for i in range(sizex):#We assign the brightest peak in each pixel to the main peak map
  for j in range(sizey):
    if (sum(mono_map[1:,i,j]) != 0):
      #index_cube[shapes[0]]=i
      #index_cube[shapes[1]]=j
      #perf=cube.__getitem__(index_cube)   
      perf=cube[:,i,j]
      v_imax=vel[numpy.where(perf==max(perf))]
      vdiff=abs(vel_map[:,i,j]-v_imax)
      nmap=numpy.where(vdiff==min(vdiff))
      
      main_v[i,j]=vel_map[nmap,i,j]#el principal esta completo
      main_i[i,j]=mono_map[nmap,i,j]
      main_d[i,j]=disp_map[nmap,i,j]
      vel_map[nmap,i,j]=0.
      mono_map[nmap,i,j]=0.
      disp_map[nmap,i,j]=0.
      
      
if in_par['MAIN_OUT'] == 1:
  main_i=(pi**0.5)*(main_i*in_par['CALIB_A']*in_par['CALIB_C']+in_par['CALIB_B']*in_par['CALIB_C']*main_d)
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
  
  #hdu_vel=pf.PrimaryHDU(main_v)
  #hdu_disp=pf.PrimaryHDU(main_d)
  #hdu_mono=pf.PrimaryHDU(main_i)
  #hdu_vel.writeto('main_v.fits',hdr_map,clobber=True)
  #hdu_disp.writeto('main_d.fits',hdr_map,clobber=True)
  #hdu_mono.writeto('main_i.fits',hdr_map,clobber=True)

if in_par['MAIN_EXT'] == 1:
  main_v=pf.open(in_par['MAIN_NAME'])[0].data


veloff=numpy.array([numpy.subtract(vel_map[k,:,:],main_v) for k in numpy.arange(1,map_number)])
veloff_abs=abs(veloff)
bub=[]
for m1 in range(map_number-2):#buscamos donde se cumplen las condiciones
  for m2 in range(m1+1,map_number-1):
     #LA SIGUIENTE LINEA NO FUNCIONA SI SE CAMBIAN NAN POR 0
    bub_m=numpy.where((abs(veloff_abs[m1,:,:]-veloff_abs[m2,:,:]) < in_par['BUB_V_SIGMA']*v_res) & (veloff_abs[m1,:,:]-veloff_abs[m2,:,:] != 0.) & (veloff_abs[m1,:,:]!=0.) & (veloff_abs[m2,:,:]!=0.) & ((veloff[m1,:,:]<0) !=(veloff[m2,:,:]<0)))# & (abs(peaksi[:,:,m1+2]-peaksi[:,:,m2+2]) < 10.) & (abs(peaksd[:,:,m1+2]-peaksd[:,:,m2+2]) < 10.))
    bub_m=[(bub_m[0][i1],bub_m[1][i1],m1,m2) for i1 in range(len(bub_m[0]))]
    bub.extend(bub_m)

if in_par['BUB_USE_I'] == 1:
  bub_i=[]
  for m1 in range(map_number-2):#buscamos donde se cumplen las condiciones
    for m2 in range(m1+1,map_number-1):
     #LA SIGUIENTE LINEA NO FUNCIONA SI SE CAMBIAN NAN POR 0
      bub_m_i=numpy.where((abs(mono_map[m1,:,:]-mono_map[m2,:,:]))/(mono_map[m1,:,:]+mono_map[m2,:,:]) < in_par['BUB_I_R'])
      bub_m_i=[(bub_m_i[0][i1],bub_m_i[1][i1],m1,m2) for i1 in range(len(bub_m_i[0]))]
      bub_i.extend(bub_m_i)
  bub=list(set(bub)&set(bub_i))


if in_par['BUB_USE_D'] == 1:
  bub_d=[]
  for m1 in range(map_number-2):#buscamos donde se cumplen las condiciones
    for m2 in range(m1+1,map_number-1):
     #LA SIGUIENTE LINEA NO FUNCIONA SI SE CAMBIAN NAN POR 0
      bub_m_d=numpy.where((abs(disp_map[m1,:,:]-disp_map[m2,:,:]))/(disp_map[m1,:,:]+disp_map[m2,:,:]) < in_par['BUB_D_R'])
      bub_m_d=[(bub_m_d[0][i1],bub_m_d[1][i1],m1,m2) for i1 in range(len(bub_m_d[0]))]
      bub_d.extend(bub_m_d)
  bub=list(set(bub)&set(bub_d))


nullmap=numpy.empty([sizex,sizey])
nullmap.fill(numpy.nan)
bubble_map=numpy.array(nullmap)#creamos el mapa con las burbujas y sacamos los perfiles
bubble_intrel_map=numpy.array(nullmap)#creamos un mapa con la intensidad relativa de los picos

bubble_sec=numpy.array(nullmap)#we create a list to contain detections of expansion that overlap with points where there si already detection (possible double bubbles)
bubble_sec_int=numpy.array(nullmap)
bubble_sec=numpy.empty([1,sizex,sizey])
bubble_sec.fill(numpy.nan)
bubble_sec_int=numpy.array(bubble_sec)

for coord in bub:
  i=coord[0]
  j=coord[1]
  m1=coord[2]
  m2=coord[3]
  voff=0.5*veloff_abs[m1,i,j]+0.5*veloff_abs[m2,i,j]
  if in_par['INT_TYPE']==0:
    irel=(mono_map[m1+1,i,j]+mono_map[m2+1,i,j])/(2.*main_i[i,j])
    #irel=(mono_map[m1+1,i,j]+mono_map[m2+1,i,j])/(2.*main_i[i,j])
  elif in_par['INT_TYPE']==1:
    irel=(mono_map[m1+1,i,j]+mono_map[m2+1,i,j])/2.
    irel=(pi**0.5)*(irel*in_par['CALIB_A']*in_par['CALIB_C']+in_par['CALIB_B']*in_par['CALIB_C']*disp_map[m2+1,i,j])
    
  if numpy.isnan(bubble_map[i,j]):
    bubble_map[i,j]=voff
    bubble_intrel_map[i,j]=irel
  else:
    if not numpy.isnan(sum(bubble_sec[:,i,j])):
      bubble_sec=list(bubble_sec)
      bubble_sec.append(nullmap)
      bubble_sec=numpy.array(bubble_sec)
    
    if voff<bubble_map[i,j]:
      bubsec_index=min(numpy.where(numpy.isnan(bubble_sec[:,0,0]))[0])
      bubble_sec[bubsec_index,i,j]=voff
      bubble_sec_int[bubsec_index,i,j]=irel
    else:
      bubsec_index=min(numpy.where(numpy.isnan(bubble_sec[:,0,0]))[0])
      bubble_sec[bubsec_index,i,j]=bubble_map[i,j]
      bubble_sec_int[bubsec_index,i,j]=bubble_intrel_map[i,j]
      bubble_map[i,j]=voff
      bubble_intrel_map[i,j]=irel



pf.writeto('bubble.fits',bubble_map,hdr_map,clobber=True)
pf.writeto('bubble_int.fits',bubble_intrel_map,hdr_map,clobber=True)

hdr_map['NAXIS']=3
hdr_map['NAXIS3']=len(bubble_sec[0,:,:])

pf.writeto('bubble_sec.fits',bubble_sec,hdr_map,clobber=True)
pf.writeto('bubble_sec_int.fits',bubble_sec_int,hdr_map,clobber=True)


if in_par['PROF_OUT'] == 1:
  try:
    os.mkdir('profiles')
  except:
    os.system('rm profiles/*')
  fig=plt.figure(1)
  for coord in bub:
    i=coord[0]
    j=coord[1]
    perf0=cube[:,i,j]
    if in_par['CENTERING'] == 1:
      perfv,perfil=line_centering(vel,perf0)
      perfvg=numpy.arange(perfv[0],perfv[-1],(perfv[1]-perfv[0])*0.1)
    else:
      perfil=perf0
      perfv=vel
      perfvg=numpy.arange(perfv[0],perfv[-1],(perfv[1]-perfv[0])*0.1)
    det_p=numpy.where(vel_map[:,i,j]!=0.)
    plt.plot(perfv,perfil,'k')
    perfil2=numpy.zeros(len(perfvg))
    for l in det_p[0]:
      perf_l=mono_map[l,i,j]*numpy.exp(-((perfvg-vel_map[l,i,j])**2.)/(2.*disp_map[l,i,j]**2.))/(disp_map[l,i,j])#c es ajustada a 2*c**2
      plt.plot(perfvg,perf_l)
      perfil2=perfil2+perf_l

    plt.plot(perfvg,perfil2,'--')
    plt.xlabel('Velocity ('+in_par['PROF_VUNIT']+')')
    plt.ylabel('I ('+in_par['PROF_IUNIT']+')')
    plt.title('x='+str(j)+', y='+str(i))#CAMBIADO DE ORDEN POR SUPOSICION DE QUE EL DS9 VA TRASPUESTO AL PYTHON
    fig.savefig('profiles/perf_x='+str(j)+'_y='+str(i))
    plt.clf()

