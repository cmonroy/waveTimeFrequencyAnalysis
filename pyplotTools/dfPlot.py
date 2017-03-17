
def dfSlider( dfList, labels = None ) :

   import matplotlib.pyplot as plt
   from matplotlib.widgets import Slider
   import numpy as np

   #Make compatible with dataFrame input
   if type(dfList) is not list : dfList = [dfList]
   if labels is None : labels = [ i for i in range(len(dfList)) ]

   fig, ax = plt.subplots()
   plt.subplots_adjust(bottom=0.20)
   ax.grid(True)
   a0 = 0
   lList = []
   for idf,df in enumerate(dfList) :
      l, = ax.plot( df.columns , df.iloc[a0,:] , lw=2 , label = labels[idf] )
      lList.append(l)

   ax.legend()

   df = dfList[0]
   ax.set_title( df.index[0] )
   plt.axis( [min(df.columns), max(df.columns), df.min().min() , df.max().max()] )

   axTime = plt.axes( [0.15, 0.10, 0.75, 0.03] , axisbg='lightgoldenrodyellow')
   sTime = Slider(axTime, 'Time', df.index[0] , df.index[-1] , valinit=a0)

   def update(val):
      for i , df in enumerate(dfList) :
         itime = np.argmin( np.abs(df.index.values - sTime.val) )
         lList[i].set_ydata( df.iloc[ itime , : ] )
         ax.set_title( df.index[ itime ] )
      fig.canvas.draw_idle()
   sTime.on_changed(update)
   plt.show()


def dfAnimate( df, movieName = None, nShaddow = 0, xRatio= 1.0, rate = 1) :
      """
         Animate a dataFrame where time is the index, and columns are the "spatial" position
      """
      from matplotlib import pyplot as plt
      from matplotlib import animation

      print "Making animation file : " , movieName

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
      ax.set_xlim( min(xVal) , max(xVal) )
      ax.set_ylim( df.min().min() , df.max().max() )
   
      ax.set_xlabel("x")
      ax.set_ylabel("Elevation(m)")
   
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


def test() :
   from Spectral.Wif import Wif
   wif = Wif.Jonswap( Hs = 2.0 , Tp = 10.0 , Gamma = 1.0 , Heading = 180. )
   wif = Wif( r"D:\Etudes\HOS\edw\my_10_rcw.wif" )
   #wif.position( -400. , 0. )
   df = wif.wave2DC( tmin = 0 , tmax = +200 , dt = 0.2 , xmin = 0. , xmax = +800 , dx = 4.0 , speed = 0.0 )
   
   df2 = wif.wave2DC( tmin = 0 , tmax = +200 , dt = 0.2 , xmin = 100. , xmax = +800 , dx = 4.0 , speed = 0.0 )

   #dfAnimate(df, nShaddow = 5, xRatio= 1.0, rate = 1)
   dfSliderComp([df,df2])


if __name__ == "__main__" :

   test()

