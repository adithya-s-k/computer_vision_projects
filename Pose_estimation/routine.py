from CounterModule import *
from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font
import time
from PIL import Image, ImageTk

root = tk.Tk(className='Main MENU')
root.geometry("620x1000")
root.minsize(620,1000)
root.maxsize(620,1000)

def exercise_callback():
    diff = tk.Tk(className='Input difficulty')
    diff.geometry("620x1000")
    diff.minsize(620,1000)
    diff.maxsize(620,1000)

    def easy():
        entry1 = 5
        diff.destroy()
        difficulty = entry1  
        print(difficulty)
        curl_counter(difficulty)
        time.sleep(5)
        running_counter(difficulty)
        time.sleep(5)
        squat_counter(1)
        cap.release()
    
    def moderate():
        entry1 = 10
        diff.destroy()
        difficulty = entry1  
        print(difficulty)
        curl_counter(difficulty)
        time.sleep(5)
        running_counter(difficulty)
        time.sleep(5)
        squat_counter(difficulty)

        cap.release()
    def hard():
        entry1 = 15
        diff.destroy()
        difficulty = entry1  
        print(difficulty)
        curl_counter(difficulty)
        time.sleep(5)
        running_counter(difficulty)
        time.sleep(5)
        squat_counter(difficulty)
        time.sleep(5)
        push_up_counter(difficulty)
        cap.release()

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

def posture_detector_callback():
    posture_detector()

def counter_time_callback():
    curl_counter(2)

L1 = ttk.Label(root, text="AIWA",font=("Helvetica",36,"bold"))
L1.place(x=310, y=30, anchor="center")

L2 = ttk.Label(root, text="Your Personalized AI workout Assistant",font=("Helvetica",16,"bold"))
L2.place(x=310, y=70, anchor="center")

L3 = ttk.Label(root, text="What do you want to do?",font=("Helvetica",18,"bold"))
L3.place(x=310, y=170, anchor="center")
s = ttk.Style()
s.configure('my.TButton', font=('Helvetica', 13 ,"bold"))

buttonPhoto = ImageTk.PhotoImage(Image.open(r"C:\Programming\Python_Projects\Image_processing_projects\Pose_estimation\program7__b.jpg"))

# MyButton1 = Button(root, text="Warm UP", image = buttonPhoto , command=exercise_callback , borderwidth=0 )
MyButton1 = ttk.Button(root, text="Warm UP", command=exercise_callback)
MyButton1.grid(row=4, column=4)
MyButton1.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=200)

MyButton2 = ttk.Button(root, text="Sitting Posture Detection", command= posture_detector_callback)
MyButton2.grid(row=4, column=4)
MyButton2.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=450)

MyButton3 = ttk.Button(root, text="Run and Jump", command=counter_time_callback)
MyButton3.grid(row=4, column=4)
MyButton3.place(bordermode=OUTSIDE, height=200, width=300,x=160,y=700)

root.mainloop()