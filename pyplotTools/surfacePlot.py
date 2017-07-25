
from matplotlib import pyplot as plt
import numpy as np

def mapFunction( x , y , func , ax = None, arrayInput = False, n = 10, title = None, **kwargs ) :
   """
      Plot function on a regular grid
        x : 1d array
        y : 1d array
        func : function to map
        arrayInput : False if func(x,y) , True if func( [x,y] )
   """

   if ax is None :
      fig , ax = plt.subplots()
      
   X , Y = np.meshgrid( x , y )

   if not arrayInput :
      Z = func( X.flatten() , Y.flatten() ).reshape(X.shape)
   else :
      Z = func( np.stack( [ X.flatten() , Y.flatten() ]) )

   ax.contourf( X , Y , Z , n , **kwargs)

   if title is not None : ax.set_title(title)

   return ax

if __name__ == "__main__" :


   def func( x , y ) : return x**2-y**2
   x = np.linspace( 0.1 , 2.0 , 10)
   y = np.linspace( 0.1 , 3.0 , 15)
   mapFunction( x , y , func )
   plt.show()

   




