#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "realft.h"

void onedfft(double *data, double *mag, double *phase, int npoints){
  int i; double *temp;

  if(!(temp=(double *) malloc(npoints*sizeof(double))))
  {
   perror("error making temp");
   exit(1);
  }
 
  for(i=0;i<npoints;i++) temp[i] = data[i];  
  realft(temp,npoints/2,1);

  /* experimentally determined that peak value of sine = FFT mag / npoints * 2. 
  */

  mag[0] = sqrt(temp[0]*temp[0]) / npoints * 2.;
  for(i=1;i<npoints/2;i++) 
    mag[i] = sqrt(temp[i+i]*temp[i+i]+temp[i+i+1]*temp[i+i+1]) / npoints * 2.;
  mag[npoints/2] = sqrt(temp[1]*temp[1]) / npoints * 2.;

  if(temp[0]<0)   phase[0]= -180;
  else phase[0] = 0.0;
  for(i=1;i<npoints/2;i++) 
    /* Note: this sign convention makes phase leads come out positive */
    phase[i]=180.*atan2(-1*temp[i+i+1],temp[i+i])/3.14159265359;
    //    phase[i]=180.*atan2(temp[i+i+1],temp[i+i])/3.14159265359;
  if(temp[1]<0) phase[npoints/2]= -180;
  else phase[npoints/2] = 0.0;
}
