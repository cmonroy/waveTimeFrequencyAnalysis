"""

   Additional plotting tools for pandas DataFrame

"""


from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



def centerToEdge( array ) :
   dx = array[1:] - array[:-1]
   if (abs(dx / dx[0] - 1)<0.01 ).all()  :
      return np.append( array - 0.5*dx[0] , array[-1] + 0.5*dx[0] )
   else :
      print ("Can not find edge from center if bins are not considered evenly spaced")
      raise(ValueError())


def dfSurface( df, labels = None ,ax = None , nbColors = 200 , interpolate = True ,
               polar = False, polarConvention = "seakeeping" , colorbar = False, **kwargs ) :
   """Surface plot from dataframe
          index : y or theta
          columns : x or theta
          data = data

      if interpolate is True, data are considered as node value and interpolated in between
      if interpolate is False, data are considered as center cell value
   """

   if ax is None :
      fig = plt.figure()
      ax = fig.add_subplot(111, polar = polar)
      if polar : 
          if polarConvention == "seakeeping" :
              ax.set_theta_zero_location("S")
              ax.set_theta_direction(1)
          elif polarConvention == "geo" : 
              ax.set_theta_zero_location("N")
              ax.set_theta_direction(1)

   if interpolate :
      cax = ax.contourf( df.columns.astype(float) , df.index , df.values, nbColors, **kwargs  )
   else :
      try :
         cax = ax.pcolormesh( centerToEdge ( df.columns.astype(float) ) , centerToEdge ( df.index ) , df.values , **kwargs  )
      except :
         raise( Exception ("Index is not evenly spaced, try with interpolate = True") )

   #Add x and y label if contains in the dataFrame
   if df.columns.name is not None :
       ax.set_xlabel( df.columns.name )
   if df.index.name is not None :
       ax.set_ylabel( df.index.name )

   # Add colorbar, make sure to specify tick locations to match desired ticklabels
   if colorbar :
      ax.get_figure().colorbar( cax )      
         
   return ax
   

def dfIsoContour( df, ax = None , polar = False, polarConvention = "seakeeping" , inline = True , **kwargs ) :
   """Iso contour plot from dataframe
          index : y or theta
          columns : x or theta
          data = data
   """
   if ax is None :
      fig = plt.figure()
      ax = fig.add_subplot(111, polar = polar)
      if polar :
          if polarConvention == "seakeeping" :
              ax.set_theta_zero_location("S")
              ax.set_theta_direction(1)
          elif polarConvention == "geo" :
              ax.set_theta_zero_location("N")
              ax.set_theta_direction(1)

   cax = ax.contour( df.columns.astype(float) , df.index , df.values, **kwargs  )

   if inline :
      ax.clabel(cax, inline=1 , fontsize=10 ,  fmt  = r"%1.1f")

   ax.legend()
   #Add x and y label if contains in the dataFrame
   if df.columns.name is not None :
       ax.set_xlabel( df.columns.name )
   if df.index.name is not None :
       ax.set_ylabel( df.index.name )
   return ax


def dfSlider( dfList, labels = None , ax = None , display = True) :
   """ Interactive 2D plots, with slider to select the frame to display
   
   Column is used as x axis
   Index is used as frame/time (which is selected with the slider)

   :param dfList: List of DataFrame to animate
   :param labels: labels default = 1,2,3...
   :param ax: Use existing ax if provided
   :param display: display the results (wait for next show() otherwise)

   :return:  ax

   """


   print ("Preparing interactive plot")
   from matplotlib.widgets import Slider
   import numpy as np

   #Make compatible with single dataFrame input
   if type(dfList) is not list : dfList = [dfList]
   if labels is None : labels = [ i for i in range(len(dfList)) ]

   if ax is None :
      fig, ax = plt.subplots()

   plt.subplots_adjust(bottom=0.20)
   ax.grid(True)

   a0 = 0
   global currentValue
   currentValue = dfList[0].index[a0]

   lList = []
   for idf,df in enumerate(dfList) :
      l, = ax.plot( df.columns , df.iloc[a0,:] , lw=2 , label = labels[idf] )
      lList.append(l)

   ax.legend( loc = 2)

   df = dfList[0]
   ax.set_title( df.index[0] )

   tmin = min( [min(df.columns) for df in dfList]  )
   tmax = max( [max(df.columns) for df in dfList]  )
   ymin = min( [df.min().min() for df in dfList]  )
   ymax = max( [df.max().max() for df in dfList]  )
   plt.axis( [tmin, tmax, ymin , ymax ] )

   axTime = plt.axes( [0.15, 0.10, 0.75, 0.03] , facecolor='lightgoldenrodyellow')
   sTime = Slider(axTime, 'Time', df.index[0] , df.index[-1] , valinit=a0)

   def update(val):
      global currentValue
      t = []
      for i , df in enumerate(dfList) :
         itime = np.argmin( np.abs(df.index.values - sTime.val) )
         lList[i].set_ydata( df.iloc[ itime , : ] )
         t.append ( "{:.1f}".format(  df.index[ itime ]) )
         currentValue = val
      ax.set_title( " ".join(t) )
      fig.canvas.draw_idle()

   update( currentValue )

   def scroll(event) :
      global currentValue
      s = 0
      if event.button == 'down' and currentValue < tmax : s = +1
      elif event.button == 'up' and currentValue > tmin : s = -1
      dt = dfList[0].index[1]-dfList[0].index[0]
      sTime.set_val(currentValue + s*dt )

   fig.canvas.mpl_connect('scroll_event', scroll )
   sTime.on_changed(update)

   if display :
      plt.show()

   #Return ax, so that it can be futher customized ( label... )
   return ax



def dfAnimate( df, movieName = None, nShaddow = 0, xRatio= 1.0, rate = 1 , xlim = None , ylim = None , xlabel = "x(m)" , ylabel = "Elevation(m)") :
      """
         Animate a dataFrame where time is the index, and columns are the "spatial" position
      """
      
      from matplotlib import animation

      print ("Making animation file : " , movieName)

      global pause
      pause = False
      def onClick(event):
          global pause
          pause ^= True

      nShaddow = max(1,nShaddow)

      fig, ax = plt.subplots()
      fig.canvas.mpl_connect('button_press_event', onClick )
      ls = []
      for i in range(nShaddow) :
         if i == 0 :
            color = "black"
         else :
            color = "blue"
         ltemp,  = ax.plot( [], [], lw=1 , alpha = 1-i*1./nShaddow , color = color)
         ls.append(ltemp)
   
      xVal = df.columns

      ax.grid(True)
      
      if xlim :
        ax.set_xlim( xlim )
      else :
        ax.set_xlim( min(xVal) , max(xVal) )

      if ylim :
        ax.set_ylim( ylim )
      else :
         ax.set_ylim( df.min().min() , df.max().max() )

      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)

      def run(itime):
        ax.set_title("{}s".format(df.index[itime*rate]) )
        for s in range(nShaddow):
           if not pause :
              if itime > s :
                 ls[s].set_data( xVal , df.iloc[ rate*(itime - s) , : ] )
        return ls

      ani = animation.FuncAnimation( fig , run, range(len(df)), blit=True, interval=30, repeat=False)

      if movieName is None :
         plt.show()
      else :
         mywriter = animation.FFMpegWriter(fps = 25 , codec="libx264")
         ani.save( movieName +'.mp4' , writer=mywriter)


def testSlider() :
   from Spectral import Wif
   wif = Wif.Jonswap( Hs = 2.0 , Tp = 10.0 , Gamma = 1.0 , Heading = 180. )
   #wif = Wif( r"D:\Etudes\HOS\edw\my_10_rcw.wif" )
   #wif.position( -400. , 0. )
   df = wif.wave2DC( tmin = 0 , tmax = +200 , dt = 0.2 , xmin = 0. , xmax = +800 , dx = 4.0 , speed = 0.0 )
   df2 = wif.wave2DC( tmin = 0 , tmax = +200 , dt = 0.2 , xmin = 100. , xmax = +800 , dx = 4.0 , speed = 0.0 )
   #dfAnimate(df, nShaddow = 5, xRatio= 1.0, rate = 1)
   dfSlider( [df,df2] )
   

def testSurfacePlot():
    df = pd.DataFrame(  index = np.linspace(0,0.5,100) , columns = np.linspace(0,2*np.pi,50), dtype = "float" )
    df.loc[:,:] = 1
    for i in range(len(df.columns)):
       df.iloc[:,i] = np.sin(df.columns[i]) * df.index.values
    ax = dfSurface(df , polar = True ,  interpolate = True, polarConvention = "geo", colorbar = True)
    ax = dfIsoContour(df , levels = [0.0 , 0.5 ] , ax = ax)
    plt.show()

if __name__ == "__main__" :

   print ("Test")
   testSurfacePlot( )
   #testSlider()

