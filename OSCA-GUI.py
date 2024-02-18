import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.stats as stats

def plot_histogram(data, usl, lsl, canvas):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    ax.axvline(x=usl, color='red', linestyle='--', linewidth=2)
    ax.text(usl, ax.get_ylim()[1], f'USL= {usl}', fontsize=12, color='r')
    ax.axvline(x=lsl, color='red', linestyle='--', linewidth=2)
    ax.text(lsl, ax.get_ylim()[1], f'LSL= {lsl}', fontsize=12, color='r')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    canvas.draw()

def plot_xbar_chart(data, canvas):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sn, ss = np.array(data).shape  # Convert data to numpy array
    print("Shape of data array:", sn, ss)  # Add this line for debugging
    means = np.mean(data, axis=1)
    sx = np.linspace(1, sn, sn)
    ax.plot(sx, means, marker='o')
    ax.set_ylabel('Sample Mean')
    ax.set_title('Xbar Chart')
    canvas.draw()

def plot_r_chart(data, canvas):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sn, ss = np.array(data).shape  # Convert data to numpy array
    ranges = [np.ptp(row) for row in data]
    sx = np.linspace(1, sn, sn)
    sx = sx.astype(int)
    ax.plot(sx, ranges, marker='o')
    ax.set_ylabel('Sample Range')
    ax.set_title('R Chart')
    canvas.draw()

def plot_last_subgroups(data, canvas):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sn, ss = np.array(data).shape  # Convert data to numpy array
    sx = np.linspace(1, sn, sn)
    sx = sx.astype(int)
    c = 0
    while c < sn:
        for i in data[c, :]:
            ax.scatter(c + 1, i, color='b', marker='+')
        c += 1
        if c == sn:
            break
    ax.set_ylabel('Values')
    ax.set_xlabel('Sample')
    ax.set_title(f'Last {sn} Subgroups')
    canvas.draw()

def plot_normal_probability_plot(data, canvas):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    flat_data = np.array(data).flatten()  # Convert data to numpy array
    result = stats.anderson(flat_data, dist='norm')
    ad = result.statistic
    sdata = np.sort(flat_data)
    pr = (np.arange(len(sdata)) + 0.5) / len(sdata)
    statistic, p = stats.shapiro(data)
    ax.grid(axis='x', which='major', linestyle='')
    ax.scatter(sdata, pr, marker='o', facecolors='none', edgecolors='blue')
    ax.set_title('Normal Probability Plot')
    ax.grid(True)
    ax.text(np.max(sdata) + np.std(data) * 5 / 4, .5,
            f'Mean: {np.round(np.mean(data), 3)} \nStandard Deviation: {np.round(np.std(data), 3)} \nN: {len(data)}\nAD:{np.round(ad, 3)}\nP-Value:{np.round(p, 3)}',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    canvas.draw()

def generate_plots():
    data = parse_data()
    if data is None:
        return
    usl = float(usl_entry.get())
    lsl = float(lsl_entry.get())
    plot_histogram(data, usl, lsl, canvas)
    plot_xbar_chart(data, canvas)
    plot_r_chart(data, canvas)
    plot_last_subgroups(data, canvas)
    plot_normal_probability_plot(data, canvas)

def parse_data():
    try:
        data_text = data_entry.get("1.0", tk.END)
        data = [float(x) for x in data_text.split()]
        return data
    except ValueError:
        tk.messagebox.showerror("Error", "Invalid data format. Please enter numeric values only.")
        return None

window = tk.Tk()
window.title("Statistical Process Control")

label = tk.Label(window, text="Paste your data here:")
label.pack()

data_entry = tk.Text(window, height=10, width=50)
data_entry.pack()

label = tk.Label(window, text="Upper Specification Limit (USL):")
label.pack()

usl_entry = tk.Entry(window)
usl_entry.pack()

label = tk.Label(window, text="Lower Specification Limit (LSL):")
label.pack()

lsl_entry = tk.Entry(window)
lsl_entry.pack()

button = tk.Button(window, text="Generate Plots", command=generate_plots)
button.pack()

canvas = FigureCanvasTkAgg(Figure(figsize=(5, 4), dpi=100), master=window)
canvas.get_tk_widget().pack()

window.mainloop()
