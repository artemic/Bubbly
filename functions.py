import pyfits as pf
import numpy
import deriv
from gaussfitter import multigaussfit

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
      

def line_centering(vel,perf):
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







def deviation(vels,ints,disps,arrv,arri,arrd):
  
  ints2=ints[numpy.where(ints !=0)]
  int_mean=numpy.mean(ints2)
  int_var=(numpy.var(ints2))
  
  vels2=vels[numpy.where(vels !=0)]
  vel_mean=numpy.mean(vels2)
  vel_var=(numpy.var(vels2))
  
  disps2=disps[numpy.where(disps !=0)]
  disp_mean=numpy.mean(disps2)
  disp_var=(numpy.var(disps2))
  fv=abs(arrv-vel_mean)/vel_mean ##desviacion respecto a la media
  fd=abs(arrd-disp_mean)/disp_mean
  fi=abs(arri-int_mean)/int_mean
  ftot=fv+fi+fd
  ftot[numpy.where(ftot==3.)]=numpy.inf
  return ftot


class profile(object): #en la clase profile defino dos metodos, get_zeros y gaussianfit
			#get_zeros me calcula la posicion de los ceros de la segunda de interes
			#gaussianfit ajusta suma de gaussianas al perfil usando las funciones de Adam Ginsburg
  def __init__(self,velocity,intensity):
    self.intensity=intensity
    self.velocity=velocity
    self.cont=0
    self.zero=[]
    self.peak=[]
  def get_zeros(self,rms2,interp):
    n_elements=len(self.velocity)*interp#numero de puntos del perfil interpolado
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
	# si es el primer cero pero la derivada segunda no va de negativo a positivo no guardo ese cero, porque 
	#corresponde a un pico anterior (que no observamos o que observamos en el otro lado del perfil 
	  if ((((self.cont+1)%2) == 0) and (self.cont != 0)):
	    velant=self.zero[-1]#busca la velocidad del anterior cero
	    ind_antvel=numpy.where(numpy.array(velocity_fino)==velant)[0]#busca el indice del perfil correspondiente

	    flux_max=max(perfil_fino[ind_antvel:i])#toma el punto mas alto entre los dos ceros de la seguna derivada
	    if (max(perfil_fino[ind[self.cont-1]:i]) >= rms2):
	      
	    #si el pico es mas alto que dos veces el rms lo considero real
	      self.zero.append(velocity_fino[i])
	      self.peak.append(flux_max)
	      ind.append(i)
	      self.cont=self.cont+1 
	    else:
	      #si no es mayor que criterio*flujo_maximo no lo cuento como cero y debo quitar el anterior
	      self.cont=self.cont-1
	      self.zero.remove(self.zero[self.cont])
	      ind.remove(ind[self.cont])
	  else:      
	    self.zero.append(velocity_fino[i])
	    
	    ind.append(i)
	    self.cont=self.cont+1 
    return self.cont
  def gaussianfit(self,num_zeros,params,limitedmin,limitedmax,minpars,maxpars):
    
    ajuste=multigaussfit(self.velocity,self.intensity,ngauss=num_zeros/2,params=params,limitedmin=limitedmin,limitedmax=limitedmax,minpars=minpars,maxpars=maxpars,quiet=True,shh=True)
    fit_params=ajuste[0][:]

    return fit_params



def header_create(hdr_cube,in_par_list):
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

  
def vel_profile(hdr,z_index,sizez,in_par):
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

