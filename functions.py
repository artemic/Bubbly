import pyfits as pf
import numpy
import deriv
from gaussfitter import multigaussfit
import matplotlib.pyplot as plt

def num_peaks(nm,aux,np):
 
  if nm == aux+np:
    return np
  else:
    return num_peaks(nm,aux+np,np+1)


def param_parser(value):
  try: 
    return int(value)
  except:
    try:
      return float(value)
    except:
      return value
      
def noise_calculator(cube,hdr_map,in_par):
  #routine to separate galaxy emission from noise
  index_cube=numpy.zeros(cube.shape)
  zlen=cube.shape[0]
  for k in range(zlen): index_cube[k,:,:]=k
  if in_par['CONT_COR']:
    zerop=numpy.min(cube,0)
    cube=cube-zerop
  ind_max=cube.argmax(axis=0)#we split the cube in two parts, a half with the emission line and a half without
  ind_out=ind_max-zlen/2
  ind_aux=ind_max+zlen/2
  ind_out[numpy.where(ind_out<0)]=ind_aux[numpy.where(ind_out<0)]
  mask_line=((abs(index_cube-ind_max)<zlen/4) | (abs(abs(index_cube-ind_max)-zlen)<zlen/4))
  mask_out=((abs(index_cube-ind_out)<zlen/4) | (abs(abs(index_cube-ind_out)-zlen)<zlen/4))
  flux_line=numpy.mean((cube)*mask_line,0)
  flux_out=numpy.mean((cube)*mask_out,0)
  mask_ratio=flux_line/flux_out>in_par['FLUX_RATIO']#we demand a minimum difference in flux between on-line and off-line
  
  std=numpy.std(cube,0)
  line_noise=numpy.median(std[numpy.where((abs(flux_line/flux_out-1)<in_par['NOISE_PERC']))])
  print 'Line noise: ',line_noise
  if in_par['CONT_COR']:
    cube=cube-line_noise
    if in_par['CONT_OUT']:
      pf.writeto('continuum.fits',zerop+line_noise,hdr_map,clobber=True)#we add the spectral noise to the calculated continumm as we are biased towards values lower than the real continuum

  maxima=numpy.max(cube,0)
  mask_max=(maxima>in_par['MAX_TO_NOISE']*line_noise)#we make an additional mask requiring a minimum maximum to noise ratio
  mask_def=mask_ratio*mask_max#the combination of both masks assures a good SNR as well as masking out emission from stars
  pf.writeto('mask.fits',mask_def*1,hdr_map,clobber=True)
  return cube,mask_def,line_noise




def line_centering(vel,perf):#center the line profile, only valid for instruments with a cyclical range like Fabry-Perots
  max_ch=numpy.where(perf==max(perf))[0][0]
  sizez=len(vel)
  v0=vel[0]
  v_res=vel[1]-vel[0]
  perf2=[]
  vel2=[]
  if max_ch < sizez/2:
    l1=sizez-sizez/2-max_ch
    l2=max_ch+sizez/2
    perf2[0:l1]=perf[l2:sizez]
    perf2[l1:sizez]=perf[0:l2]
    vel2[0:l1]=numpy.arange(-l1,0)*v_res+v0
    vel2[l1:sizez]=vel[0:sizez-l1]
    perf2=numpy.array([perf2])[0]
    vel2=numpy.array([vel2])[0]
  elif max_ch > sizez/2:
    l1=max_ch-sizez/2
    l2=sizez-max_ch
    perf2[0:(sizez/2+l2)]=perf[l1:sizez]
    perf2[(sizez/2+l2):sizez]=perf[0:l1]
    vel2[0:sizez-l1]=vel[l1:sizez]
    vel2[sizez-l1:sizez]=numpy.arange(1,l1+1)*v_res+vel[sizez-1]
    perf2=numpy.array([perf2])[0]
    vel2=numpy.array([vel2])[0]
  else:
    perf2=perf
    vel2=vel

  return vel2,perf2



def get_pars(profile_i,num_zeros,v_res):#obtain the estimated parameters for the fit, alogn with the lower and upper limits
  params=[]
  for zeros in range(num_zeros/2):
    params.append(profile_i.peak[zeros])
    params.append(0.5*(profile_i.zero[2*zeros]+profile_i.zero[2*zeros+1]))
    params.append(0.5*(profile_i.zero[2*zeros+1]-profile_i.zero[2*zeros]))
  params=numpy.array(params)
  minpars=numpy.empty(len(params))
  maxpars=numpy.empty(len(params))
  for index in range(num_zeros/2):
    minpars[3*index]=0.0#minimum intensity is 0
    minpars[3*index+1]=profile_i.zero[2*index]#minimum velocity is first zero of the second derivative
    minpars[3*index+2]=0.5*params[3*index+2]#minimum dispersion is half the distance between zeros
    
    maxpars[3*index]=1.5*params[3*index]#maximum intensity is 1.5 times the estimated value
    maxpars[3*index+1]=profile_i.zero[2*index+1]#maximum velocity is the second zero of the second derivative
    maxpars[3*index+2]=2.*params[3*index+2]#maximum dispersion is double the distance between zeros
    
    if params[3*index+2]<v_res:#if the estimated velocity dispersion is below the resolution we force values based on it
      params[3*index+2]=v_res
      minpars[3*index+2]=v_res/2.
      maxpars[3*index+2]=2*v_res
  return params,minpars,maxpars

def fix_primpars(params,minpars,maxpars,fit_params,num_zeros2,ind_max,v_res):#add the main peak parameters for the second pass, and if it has been detected, correct them
  vels=[params[index] for index in numpy.arange(1,len(params),3)]
  prin_index=min(range(len(vels)), key=lambda k: abs(vels[k]-fit_params[3*ind_max+1]))
  if abs(vels[prin_index]-fit_params[3*ind_max+1])<2.*v_res:
    num_zeros=num_zeros2
    params[3*prin_index:3*prin_index+3]=fit_params[3*ind_max:3*ind_max+3]#we impose the initial parameters of the main peak as the result of the previous fit
    
    minpars[3*prin_index]=0.8*fit_params[3*ind_max]
    minpars[3*prin_index+1]=fit_params[3*ind_max+1]-2*v_res
    minpars[3*prin_index+2]=0.7*fit_params[3*ind_max+2]
    
    maxpars[3*prin_index]=1.5*fit_params[3*ind_max]
    maxpars[3*prin_index+1]=fit_params[3*ind_max+1]+2*v_res
    maxpars[3*prin_index+2]=1.3*fit_params[3*ind_max+2]
  else:
    num_zeros=num_zeros2+2
    prin_pars=numpy.array([fit_params[3*ind_max],fit_params[3*ind_max+1],fit_params[3*ind_max+2]])
    prin_minpars=numpy.array([0.8*fit_params[3*ind_max],fit_params[3*ind_max+1]-2*v_res,0.7*fit_params[3*ind_max+2]])
    prin_maxpars=numpy.array([1.5*fit_params[3*ind_max],fit_params[3*ind_max+1]+2*v_res,1.3*fit_params[3*ind_max+2]])
    
    params=numpy.append(params,prin_pars)
    minpars=numpy.append(minpars,prin_minpars)
    maxpars=numpy.append(maxpars,prin_maxpars)
  
  return params,minpars,maxpars,num_zeros



def append_map(mapset,newmap):#function to add a blank array to a set of arrays
  mapset=list(mapset)
  mapset.append(newmap)
  mapset=numpy.array(mapset)
  return mapset


class profile(object):#class that contains the information and operations related to the line profile Ginsburg
  def __init__(self,velocity,intensity):
    self.intensity=intensity
    self.velocity=velocity
    self.cont=0
    self.zero=[]
    self.peak=[]
  def get_zeros(self,rms2,interp):
    n_elements=len(self.velocity)*interp#number of elements of the interpolated profile
    velocity_min=min(self.velocity)
    velocity_max=max(self.velocity)
    velocity_fino=(velocity_min+(velocity_max-velocity_min)*numpy.arange(n_elements)/n_elements)
    ind=[]
  
    perfil_fino=numpy.interp(velocity_fino,self.velocity,self.intensity)
    deriv1=deriv.deriv(self.intensity)
    deriv2_nointerpol=deriv.deriv(self.velocity,deriv1)
    deriv2=numpy.interp(velocity_fino,self.velocity,deriv2_nointerpol)
    self.cont=0

    for i in range(n_elements-1):
      signo=deriv2[i]*deriv2[i+1]
      if signo < 0:
	
	if not((self.cont== 0) and (deriv2[i] < deriv2[i+1])):
	  #we keep the zero only if it goes from negative to positive, as otherwise it corresponds to a different, undetected peak
	  if ((((self.cont+1)%2) == 0) and (self.cont != 0)):
	    velant=self.zero[-1]#busca la velocidad del anterior cero
	    ind_antvel=numpy.where(numpy.array(velocity_fino)==velant)[0]#busca el indice del perfil correspondiente
	    flux_max=max(perfil_fino[ind_antvel:i])#we estimate the intensity as the highest point between the zeros
	    if (max(perfil_fino[ind[self.cont-1]:i]) >= rms2):
	      
	    #we accept the peak if it is higher thatn twice the rms
	      self.zero.append(velocity_fino[i])
	      self.peak.append(flux_max)
	      ind.append(i)
	      self.cont=self.cont+1 
	    else:
	      #if not, we do not consider it and we must erase the previous zero
	      self.cont=self.cont-1
	      self.zero.remove(self.zero[self.cont])
	      ind.remove(ind[self.cont])
	  else:      
	    self.zero.append(velocity_fino[i])
	    
	    ind.append(i)
	    self.cont=self.cont+1 
    return self.cont
  def gaussianfit(self,num_zeros,params,limitedmin,limitedmax,minpars,maxpars):#wrapper for the fitting algorithm
    
    ajuste=multigaussfit(self.velocity,self.intensity,ngauss=num_zeros/2,params=params,limitedmin=limitedmin,limitedmax=limitedmax,minpars=minpars,maxpars=maxpars,quiet=True,shh=True)
    fit_params=ajuste[0][:]
    return fit_params



def header_create(hdr_cube,in_par_list):#create an appropiate header for the maps
  hdr_cube['NAXIS']=2
  hdr_cube['BITPIX']=-64
  hdr_cube['EXTEND']=True
  hdr_dict=hdr_cube.ascard
  gen=['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND','RADESYS','CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CROTA2','CDELT1','CDELT2','EQUINOX','CD1_1','CD1_2','CD2_1','CD2_2']
  hdr_map=pf.Header()

  for key in gen:
    if key in hdr_cube:
      hdr_map.append(hdr_dict[key])

  hdr_map.add_comment('')
  hdr_map.add_comment('BUBBLY parameters used to produce this file:')
  hdr_map.add_comment('')

  for line in in_par_list: hdr_map.add_comment(line[0:line.find('#')])
  
  return hdr_map


def header_correct(hdr_prev,hdr_new):#creates the correct header for the bubble maps from the relevant parameters of the previous and current run
  changekeys=['RERUN','MAIN_OUT','MAIN_EXT','MAIN_NAME','BUB_V_SIGMA','BUB_USE_I','BUB_I_R','BUB_USE_D','BUB_D_R','PROF_OUT','PROF_VUNIT','PROF_IUNIT']
  for key in changekeys:
    for k in range(23,len(hdr_prev)):#23 is to take only the comments
      line=hdr_prev[k]
      if line.startswith(key):
	hdr_prev[k]=hdr_new[k]
  
  return hdr_prev
  
def vel_profile(hdr,sizez,in_par):#produce a velocity profile from the spectral information
  vel=hdr['CRVAL3']+hdr['CDELT3']*numpy.arange(0.,float(sizez),1)
  v_res=hdr['CDELT3']

  if in_par['Z_TYPE'] == 0:
    if hdr['CTYPE3'] != 'VELOCITY':
      print 'The spectral direction is not in velocity. Check cube and parameter file to ensure all is correct'
      print 'The z-direction is in', hdr['CTYPE3'], hdr['CUNIT3']
    vel=vel*in_par['VEL_SCALE']
    v_res=vel[1]-vel[0]
  elif in_par['Z_TYPE'] == 1:
    if hdr['CTYPE3'] != 'FREQ':
      print 'The spectral direction is not in frequency. Check cube and parameter file to ensure all is correct'
      print 'The z-direction is in', hdr['CTYPE3'], hdr['CUNIT3']
    c=299792.0*in_par['VEL_SCALE']
    wave_l=((c*1.e13)/(vel*in_par['FREQ_SCALE']))
    lambda0=in_par['LAMBDA_R']
    vel=c*(wave_l**2-lambda0**2)/(wave_l**2+lambda0**2)
    v_res=vel[1]-vel[0]
  else:
    if hdr['CTYPE3'] == 'VELOCITY' or hdr['CTYPE3'] == 'FREQ':
      print 'The spectral direction is incorrectly defined. Check cube and parameter file to ensure all is correct'
      print 'The z-direction is in', hdr['CTYPE3'], hdr['CUNIT3']
      print 'assuming z-direction in wavelength'
    c=299792.0*in_par['VEL_SCALE']
    lambda0=in_par['LAMBDA_R']
    wave_l=vel*in_par['LAMBDA_SCALE']
    vel=c*(wave_l**2-lambda0**2)/(wave_l**2+lambda0**2)
    v_res=vel[1]-vel[0]
    
  return vel,v_res



def plot_profile(cube,vel,i,j,centering,vel_map,mono_map,disp_map,in_par):
  fig=plt.figure(1)
  perf=cube[:,i,j]
  if centering == 1:
    vel,perf=line_centering(vel,perf)
  vel_gauss=numpy.arange(vel[0],vel[-1],(vel[1]-vel[0])*0.1)
  det_p=numpy.where(vel_map[:,i,j]!=0.)
  plt.plot(vel,perf,'k')
  perfil2=numpy.zeros(len(vel_gauss))
  for l in det_p[0]:
    perf_l=numpy.pi**(-0.5)*mono_map[l,i,j]*numpy.exp(-((vel_gauss-vel_map[l,i,j])**2.)/(2.*disp_map[l,i,j]**2.))/(disp_map[l,i,j])#c es ajustada a 2*c**2
    plt.plot(vel_gauss,perf_l)
    perfil2=perfil2+perf_l
  plt.plot(vel_gauss,perfil2,'--')
  plt.xlabel('Velocity ('+in_par['PROF_VUNIT']+')')
  plt.ylabel('I ('+in_par['PROF_IUNIT']+')')
  plt.title('x='+str(j)+', y='+str(i))#inverted i and j as per Python notation
  fig.savefig('profiles/perf_x='+str(j)+'_y='+str(i))
  plt.clf()

