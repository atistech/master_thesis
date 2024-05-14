import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from src.nn_search_engine import NNSearchEngine
import threading
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

def combobox(root, text_variable, values, row, column):
    combobox = ttk.Combobox(root, textvariable=text_variable, values=values, state="readonly")
    combobox.set(values[0])
    combobox.grid(row=row, column=column, padx=5, pady=10)

def addResultsToTreeview(count, result1, result2):
    for result in result1:
        values_to_insert = [count, result.accuracy, result.loss, result.val_accuracy, result.val_loss, result.toString()]
        tree2.insert("", 'end', values=values_to_insert)
    values_to_insert = [count, result2.accuracy, result2.loss, result2.val_accuracy, result2.val_loss, result2.toString()]
    tree1.insert("", 'end', values=values_to_insert)

def startSearchThreading():
    threading.Thread(target=startSearch).start()

def startSearch():
    param = {
        "dataset": datasetSelection_value.get(),
        "populationSize": populationSize_value.get(),
        "maxGenerationCount": 2,
        "IsRegression": False
    }
    answer = messagebox.askokcancel(message=param)

    if answer:
        startButton.config(state="disabled")
        tree1.delete(*tree1.get_children())
        
        generationCount = 0
        search_engine = NNSearchEngine(param)
        
        for iteration_models in search_engine:
            for model in iteration_models:
                addResultsToTreeview(search_engine.generationCount, fitnessResults, populationResult)


        startButton.config(state="active")

window = tk.Tk()
window.resizable(False, False)
window.title("Neural Network Model Search Engine")

parameters_frame = tk.Frame(window)
parameters_frame.grid(row=0, column=0, padx=10, pady=10)

### Main Control Frame ###
top_frame = tk.Frame(parameters_frame)
top_frame.pack(fill='both', pady=60)
ttk.Label(top_frame, text="Neural Network Model Search Engine", font=("Arial", 14)).pack(anchor="center", pady=10)

### Genetic Algorithm Parameters Frame ###
ga_frame = tk.LabelFrame(parameters_frame, text="Genetic Algorithm Parameters")
ga_frame.pack(fill='x')
populationSize_value = tk.StringVar()
ttk.Label(ga_frame, text="Population Size:").grid(row=0, column=0, padx=5, pady=10)
combobox = ttk.Combobox(ga_frame, textvariable=populationSize_value, values=[i for i in range(2,11,2)], state="readonly")
combobox.current(4)
combobox.grid(row=0, column=1, padx=5, pady=10)
ttk.Label(ga_frame).grid(row=0, column=2, padx=20, pady=10)
maxGenerationCount_value = tk.StringVar()
ttk.Label(ga_frame, text="Max Generation Count:").grid(row=0, column=3, padx=5, pady=10)
combobox = ttk.Combobox(ga_frame, textvariable=maxGenerationCount_value, values=[i for i in range(1,11)], state="readonly")
combobox.current(9)
combobox.grid(row=0, column=4, padx=5, pady=10)

### Dataset Parameters Frame ###
dataset_frame = tk.LabelFrame(parameters_frame, text="Dataset Parameters")
dataset_frame.pack()
datasetSelection_value = tk.StringVar()
def select_file():
    filetypes = (
        ('CSV files', '*.csv'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    datasetSelection_value.set(filename)
    showinfo(
        title='Selected File',
        message=filename
    )
# open button
ttk.Button(dataset_frame, text='Open a File', command=select_file).grid(row=1, column=1, padx=20, pady=10)

startButton = tk.Button(top_frame, text="Start Search", font=("Arial", 10), command=startSearchThreading)
startButton.pack(padx=5, pady=5)

def selectItem(a):
    curItem = tree1.focus()
    item = tree1.item(curItem)
    messagebox.showinfo(message=str(item))

def createResultsTreeview(root):
    sub_frame = tk.Frame(root)
    sub_frame.pack(padx=10, pady=10)
    
    tree = ttk.Treeview(sub_frame, show="headings", selectmode="browse")
    tree["columns"] = ("1", "2", "3", "4", "5", "6")
    for i in range(6):
        if i==6:
            tree.column(str(i), width = 300, anchor ='w')
        else:
            tree.column(str(i), width = 80, anchor ='c')
    tree.heading("1", text ="Generation")
    tree.heading("2", text ="Fitness Score")
    tree.heading("3", text ="Model History")
    tree.heading("4", text ="Model Architecture")

    tree_scroll_y = tk.Scrollbar(sub_frame, orient="vertical", command=tree.yview)
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = tk.Scrollbar(sub_frame, orient="horizontal", command=tree.xview)
    tree_scroll_x.pack(side="bottom", fill="x")
    
    tree.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
    tree.pack()
    tree.bind('<ButtonRelease-1>', selectItem)
    return tree

best_results_frame = tk.LabelFrame(parameters_frame, text="Search Results")
best_results_frame.pack()
tree1 = createResultsTreeview(best_results_frame)

ttk.Label(parameters_frame, text="Developed by Atakan Şentürk").pack(anchor='w', pady=10)

window.mainloop()