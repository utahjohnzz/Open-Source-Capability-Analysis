import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_histogram(data,lsl,usl):
    
    data = [float(value) for value in data]
    
    
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.5, color='blue', edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    plt.axvline(x=usl, color='red', linestyle='--', linewidth=2)
    plt.text(usl,plt.gca().get_ylim()[1], f'USL= {usl}', fontsize=12, color='r')
    plt.axvline(x=lsl, color='red', linestyle='--', linewidth=2)
    plt.text(lsl,plt.gca().get_ylim()[1], f'LSL= {lsl}', fontsize=12, color='r')
    
    
    # Add a normal line
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 100)
    p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/std)**2)
    ax.plot(x, p, 'k', linewidth=2)
    
    return fig

def get_nvalue():
    global N, usl, lsl
    N = float(entry_nvalue.get())
    usl = float(entry_usl.get())
    lsl = float(entry_lsl.get())
    root.destroy()  
    
    
    
    
    
    
    
    
    
    
    
    
    data_window = tk.Tk()
    data_window.title("Data Input")
    
    
    frame = ttk.Frame(data_window)
    frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
    
    
    
    
    
    
    
    ttk.Label(frame, text="Paste {}x1 array data:".format(N)).pack()
    data_input = tk.Text(frame, height=10, width=20)
    data_input.pack(side=tk.LEFT, padx=5)
    
    def get_data():
        
        data = data_input.get("1.0", tk.END).strip().split("\n")
        
        fig = plot_histogram(data,lsl,usl)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, padx=5)
    
    ttk.Button(data_window, text="Get Data", command=get_data).pack()
    
    data_window.mainloop()


root = tk.Tk()
root.title("Number of Values Input")


ttk.Label(root, text="Enter number of values, N:").pack()
entry_nvalue = ttk.Entry(root)
entry_nvalue.pack(pady=10)





ttk.Label(root, text="Enter Upper Specification Limit, USL:").pack()
entry_usl = ttk.Entry(root)
entry_usl.pack(pady=10)



ttk.Label(root, text="Enter Lower Specification Limit, LSL:").pack()
entry_lsl = ttk.Entry(root)
entry_lsl.pack(pady=10)

get_lsl_button = ttk.Button(root, text="Continue", command=get_nvalue)
get_lsl_button.pack()

root.mainloop()


