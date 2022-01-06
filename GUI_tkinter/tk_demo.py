from tkinter import *
from screeninfo import get_monitors

for m in get_monitors():
    print(str(m))

gui = Tk(className='Python Examples - Window Size')
# set window size
gui.geometry("620x1000")
# set minimum window size value
gui.minsize(620,1000)
# set maximum window size value
gui.maxsize(620,1000)

gui.mainloop() 