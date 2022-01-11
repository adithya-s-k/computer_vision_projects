from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import time

root = tk.Tk(className='Main MENU')
root.geometry("620x1000")
root.minsize(620,1000)
root.maxsize(620,1000)


def options():
    diff = tk.Tk(className='Input difficulty')
    diff.geometry("620x1000")
    diff.minsize(620,1000)
    diff.maxsize(620,1000)

    def easy():
        global entry1
        entry1 = 5
        diff.destroy()
    
    def moderate():
        global entry1
        entry1 = 10
        diff.destroy()
    
    def hard():
        global entry1
        entry1 = 15
        diff.destroy()

    L1 = ttk.Label(diff, text="AIWA",font=("Helvetica",36,"bold"))
    L1.place(x=310, y=30, anchor="center")

    L2 = ttk.Label(diff, text="Your Personalized AI workout Assistant",font=("Helvetica",16,"bold"))
    L2.place(x=310, y=70, anchor="center")

    L3 = ttk.Label(diff, text="Choose A Difficulty Level",font=("Helvetica",18,"bold"))
    L3.place(x=310, y=170, anchor="center")
    # E1 = tk.Entry(diff, bd = 5)
    # E1.grid(row=0, column=1)

    MyButton1 = ttk.Button(diff, text="EASY", command= easy)
    MyButton1.grid(row=4, column=4)
    MyButton1.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=200)

    MyButton2 = ttk.Button(diff, text="MODERATE",command= moderate)
    MyButton2.grid(row=4, column=4)
    MyButton2.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=450)

    MyButton3 = ttk.Button(diff, text="HARD", command=hard)
    MyButton3.grid(row=4, column=4)
    MyButton3.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=700)
    

    diff.mainloop()

L1 = ttk.Label(root, text="AIWA",font=("Helvetica",36,"bold"))
L1.place(x=310, y=30, anchor="center")

L2 = ttk.Label(root, text="Your Personalized AI workout Assistant",font=("Helvetica",16,"bold"))
L2.place(x=310, y=70, anchor="center")

L3 = ttk.Label(root, text="What do you want to do?",font=("Helvetica",18,"bold"))
L3.place(x=310, y=170, anchor="center")
# E1 = tk.Entry(root, bd = 5)
# E1.grid(row=0, column=1)

MyButton1 = ttk.Button(root, text="Full Body Warm UP", command=options)
MyButton1.grid(row=4, column=4)
MyButton1.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=200)

MyButton2 = ttk.Button(root, text="Sitting Posture Detection")#, command=callback)
MyButton2.grid(row=4, column=4)
MyButton2.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=450)

MyButton3 = ttk.Button(root, text="Run and Jump")#, command=callback)
MyButton3.grid(row=4, column=4)
MyButton3.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=700)

root.mainloop()

difficulty = entry1  
print(difficulty)
