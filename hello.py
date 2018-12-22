from tkinter import *
from tkinter import ttk
gui = Tk()
gui.geometry("400x400")
gui.title("Axis AI Challenge")

def new_winF(): # new window definition
    newwin = Toplevel(gui)
    display = Label(newwin, text="Humm, see a new window !")
    display.pack()    

user = Label(gui, text = "username").pack()
user_entry = Entry(gui).pack(ipady=3)

pasw = Label(gui ,text="password").pack()
pasw_entry = Entry(gui, show="*").pack(ipady=3)

login = Button(gui ,text="Login", command = new_winF).pack(pady=10)

sign = Button(gui ,text="Sign Up").pack()

gui.mainloop()