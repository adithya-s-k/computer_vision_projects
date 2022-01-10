from tkinter import *
import tkinter as tk
from CounterModule import *
import time

root = Tk(className='Input difficulty')
root.geometry("620x1000")
root.minsize(620,1000)
root.maxsize(620,1000)

def callback():
    global entry1
    entry1 = E1.get()
    root.destroy()


L1 = Label(root, text="User Name")
L1.grid(row=0, column=0)
E1 = tk.Entry(root, bd = 5)
E1.grid(row=0, column=1)
MyButton1 = Button(root, text="Submit", width=10, command=callback)
MyButton1.grid(row=1, column=1)

root.mainloop()

difficulty = entry1  

curl_counter(difficulty)
time.sleep(2)
squat_counter(difficulty)
cap.release()