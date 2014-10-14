#!/usr/bin/python -tt
#=======================================================================
#                        General Documentation

"""Single-function module.

   See function docstring for description.
"""

#-----------------------------------------------------------------------
#                       Additional Documentation
#
# RCS Revision Code:
#   $Id: deriv.py,v 1.3 2004/03/17 00:51:34 jlin Exp $
#
# Modification History:
# - 22 Nov 2003:  Original by Johnny Lin, Computation Institute,
#   University of Chicago.  Passed passably reasonable tests.
#
# Notes:
# - Written for Python 2.2.
# - Module docstrings can be tested using the doctest module.  To
#   test, execute "python deriv.py".
# - See import statements throughout for non-"built-in" packages and
#   modules required.
#
# Copyright (c) 2003 by Johnny Lin.  For licensing, distribution 
# conditions, contact information, and additional documentation see
# the URL http://www.johnny-lin.com/py_pkgs/gemath/doc/.
#=======================================================================




#----------------------- Overall Module Imports ------------------------

#- Set module version to package version:

#import gemath_version
#__version__ = gemath_version.version
#del gemath_version




#---------------------- General Function:  deriv -----------------------

def deriv(*positional_inputs, **keyword_inputs):
    """Calculate the derivative along a single dimension.


    Calling Sequence:
        Result = deriv([x_in,] y_in, missing=1e+20, algorithm='default')


    Positional Input Arguments:
    * x_in:  Abscissa values of y_in to take with respect to.  If 
      not set, the derivative of y_in is take with respect to unit 
      abscissa intervals.  Numeric array of same shape and size as 
      y_in.  Must be monotonic and with no duplicate values.
      Optional.  First positional argument out of two, if present.

    * y_in:  Ordinate values, to take the derivative with respect 
      to.  Numeric array vector of rank 1.  Required.  Second posi-
      tional argument, if x_in is present; only positional argument 
      if x_in is absent.


    Keyword Input Arguments:
    * missing:  If y_in and/or x_in has missing values, this is the 
      missing value value.  Scalar.  Default is 1e+20.

    * algorithm:  Name of the algorithm to use.  String scalar.
      Default is 'default'.  Possible values include:
      + 'default':  Default method (currently set to 'order1').
      + 'order1':  First-order finite-differencing (backward and
        forward differencing used at the endpoints, and centered
        differencing used everywhere else).  If abscissa intervals
        are irregular, differencing will be correspondingly asym-
        metric.


    Output Result:
    * Derivative of y_in with respect to x_in (or unit interval 
      abscissa, if x_in is not given).  Numeric array of same shape 
      and size as y_in.  If there are missing values, those elements 
      in the output are set to the value in |missing|.  For instance, 
      if y_in is only one element, a one-element vector is returned 
      as the derivative with the value of |missing|.  If there are 
      missing values in the output due to math errors and |missing| 
      is set to None, output will fill those missing values with the 
      MA default value of 1e+20.  


    References:
    * Press, W. H., et al. (1992):  Numerical Recipes in Fortran 
      77:  The Art of Scientific Computing.  New York, NY:  Cambridge
      University Press, pp. 180-184.

    * Wang, Y. (1999):  "Numerical Differentiation," Introduction to 
      MHD Numerical Simulation in Space, ESS265: Instrumentation, 
      Data Processing and Data Analysis in Space Physics (UCLA).
      URL:  http://www-ssc.igpp.ucla.edu/personnel/russell/ESS265/
      Ch10/ylwang/node21.html.


    Example with one argument, no missing values, using the default
    method:
    >>> from deriv import deriv
    >>> import Numeric as N
    >>> y = N.sin(N.arange(8))
    >>> dydx = deriv(y)
    >>> ['%.7g' % dydx[i] for i in range(4)]
    ['0.841471', '0.4546487', '-0.3501755', '-0.83305']
    >>> true = N.cos(N.arange(8))  #- Compare with exact solution
    >>> ['%.7g' % true[i] for i in range(4)]  
    ['1', '0.5403023', '-0.4161468', '-0.9899925']

    Example with two arguments with missing values, using first-
    order differencing:
    >>> x = N.arange(8)/(2.*N.pi)
    >>> y = N.sin(x)
    >>> y[3] = 1e20            #- Set an element to missing value
    >>> dydx = deriv(x, y, missing=1e20, algorithm='order1')
    >>> ['%.7g' % dydx[i] for i in range(5)]
    ['0.9957836', '0.9831985', '1e+20', '0.8844179', '1e+20']
    >>> true = N.cos(x)       #- Compare with exact solution
    >>> ['%.7g' % true[i] for i in range(5)]  
    ['1', '0.9873616', '0.9497657', '0.8881628', '0.8041098']
    """
    import numpy.ma as MA
    import numpy as N


    #- Establish y_in and x_in from *positional_inputs:

    if len(positional_inputs) == 1:
        y_in = positional_inputs[0]
        x_in = N.arange(len(y_in))
    elif len(positional_inputs) == 2:
        x_in = positional_inputs[0]
        y_in = positional_inputs[1]
    else:
        raise ValueError, "deriv:  Bad inputs"


    #- Establish missing and algorithm from *keyword_inputs:

    if keyword_inputs.has_key('missing') == 1:
        missing = keyword_inputs['missing']
    else:
        missing = 1e+20

    if keyword_inputs.has_key('algorithm') == 1:
        algorithm = keyword_inputs['algorithm']
    else:
        algorithm = 'default'


    #- Check positional and keyword inputs for possible errors:

    if (len(y_in.shape) != 1) or (len(x_in.shape) != 1):
        raise ValueError, "deriv:  Inputs not a vector"
    if type(algorithm) != type(''):
        raise ValueError, "deriv:  algorithm not str"


    #- Set algorithm_to_use variable, based on the algorithm keyword.
    #  The algorithm_to_use tells which algorithm below to actually
    #  use (so here is where we set what algorithm to use for default):

    if algorithm == 'default':
        algorithm_to_use = 'order1'
    else:
        algorithm_to_use = algorithm


    #- Change input to MA:  just set to input value unless there are
    #  missing values, in which case add mask:

    if missing == None:
        x = MA.masked_array(x_in)
        y = MA.masked_array(y_in)
    else:
        x = MA.masked_values(x_in, missing, copy=0)
        y = MA.masked_values(y_in, missing, copy=0)


    #- Calculate and return derivative:

    #  * Create working arrays that are consistent with a 3-point
    #    stencil in the interior and 2-point stencil on the ends:
    #    *im1 means the point before index i, *ip1 means the point 
    #    after index i, and the i index array is just plain x or 
    #    y; the endpadded arrays replicate the ends of x and y.
    #    I use an MA array filled approach instead of concatentation
    #    because the MA concatenation routine doesn't work right
    #    when the endpoint element is a missing value:

    x_endpadded = MA.zeros(len(x)+2)
    x_endpadded[0]    = x[0]
    x_endpadded[1:-1] = x 
    x_endpadded[-1]   = x[-1]

    y_endpadded = MA.zeros(len(y)+2)
    y_endpadded[0]    = y[0]
    y_endpadded[1:-1] = y
    y_endpadded[-1]   = y[-1]

    y_im1 = y_endpadded[:-2]
    y_ip1 = y_endpadded[2:]
    x_im1 = x_endpadded[:-2]
    x_ip1 = x_endpadded[2:]


    #  * Option 1:  First-order differencing (interior points use
    #    centered differencing, and end points use forward or back-
    #    ward differencing, as applicable):

    if algorithm_to_use == 'order1':
        dydx = (y_ip1 - y_im1) / (x_ip1 - x_im1) 


    #  * Option 2:  Bad algorithm specified:

    else:
        raise ValueError, "deriv:  bad algorithm"


    #- Return derivative as Numeric array:

    return MA.filled( dydx, missing )




#-------------------------- Main:  Test Module -------------------------

#- Define additional examples for doctest to use:

__test__ = {'Additional Examples':
    """
    >>> from deriv import deriv
    >>> import Numeric as N
    >>> y = N.array([3.,4.5])
    >>> dydx = deriv(y)
    >>> ['%.7g' % dydx[i] for i in range(len(dydx))]
    ['1.5', '1.5']
    >>> y = N.array([3.])
    >>> dydx = deriv(y, missing=1e+25)
    >>> ['%.7g' % dydx[i] for i in range(len(dydx))]
    ['1e+25']
    """}


#- Execute doctest if module is run from command line:

if __name__ == "__main__":
    """Test the module.

    Tests the examples in all the module documentation strings, plus
    __test__.

    Note:  To help ensure that module testing of this file works, the
    parent directory to the current directory is added to sys.path.
    """
    import doctest, sys, os
    sys.path.append(os.pardir)
    doctest.testmod(sys.modules[__name__])




# ===== end file =====
