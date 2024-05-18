import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from src.nn_search_engine import NNSearchEngine
import threading
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

s = ttk.Style()
s.configure("text.TRadioButton", font=("Helvetica", 12))

def combobox(root, text_variable, values, row, column):
    combobox = ttk.Combobox(root, textvariable=text_variable, values=values, state="readonly")
    combobox.set(values[0])
    combobox.grid(row=row, column=column, padx=5, pady=0)

def startSearchThreading():
    if(datasetSelection_value.get() == ""):
        messagebox.showwarning(title="Warning", message="Please, firstly select a csv file!!")
    else:
        threading.Thread(target=startSearch).start()

def startSearch():
    param = {
        "dataset": datasetSelection_value.get(),
        "populationSize": populationSize_value.get(),
        "maxGenerationCount": maxGenerationCount_value.get(),
        "IsRegression": True if problemType.get()==1 else False
    }
    answer = messagebox.askokcancel(message=param)

    if answer:
        startSearchButton.config(state="disabled")
        tree.delete(*tree.get_children())
        
        search_engine = NNSearchEngine(param)
        
        for iteration_models in search_engine:
            for model in iteration_models:
                model = model.toDict()
                values_to_insert = [search_engine.generationCount, model["fitnessScore"], model["architecture"], model["history"]]
                tree.insert("", 'end', values=values_to_insert)
            level.set(int(100*search_engine.generationCount/int(maxGenerationCount_value.get())))
        
        messagebox.showinfo(title="Info", message="Searching has finished.")
        startSearchButton.config(state="active")
        level.set(0)

window = tk.Tk()
#window.resizable(False, False)
window.title("Neural Network Search Engine")

### Main Control Frame ###
ttk.Label(window, text="Neural Network Search Engine", font=("Helvetica", 14)).pack(anchor="center", pady=10)

### Genetic Algorithm Parameters Frame ###
ga_frame = tk.LabelFrame(window, text="Parameters Selection")
ga_frame.pack(fill='x', padx=10, pady=10)

populationSize_value = tk.StringVar()
ttk.Label(ga_frame, text="Population Size:", font=("Helvetica", 10)).grid(row=1, column=0)
combobox(ga_frame, populationSize_value, [*range(4,11,2)], row=1, column=1)

maxGenerationCount_value = tk.StringVar()
ttk.Label(ga_frame, text="Max Generation Count:", font=("Helvetica", 10)).grid(row=1, column=3)
combobox(ga_frame, maxGenerationCount_value, [*range(1,11)], row=1, column=4)

datasetSelection_value = tk.StringVar()
ttk.Label(ga_frame, text="Dataset:").grid(row=0, column=0, padx=5, pady=10)
def select_file():
    filename = fd.askopenfilename(
        title="Open CSV File",
        filetypes=(
            ("CSV files", "*.csv"),
        )
    )
    datasetSelection_value.set(filename)
ttk.Button(ga_frame, text="Open CSV File", command=select_file).grid(row=0, column=1, padx=5, pady=10)
#ttk.Label(ga_frame, textvariable=datasetSelection_value).grid(row=0, column=2, padx=10, pady=10)

problemTypeFrame = tk.LabelFrame(ga_frame, text="Problem Type")
problemTypeFrame.grid(row=2, column=0)
problemType = tk.IntVar()
print(ttk.Radiobutton(problemTypeFrame, text="Classifciation", variable=problemType, value=2).winfo_class())
ttk.Radiobutton(problemTypeFrame, text="Regression", variable=problemType, value=1, style="text.TRadioButton").pack()
ttk.Radiobutton(problemTypeFrame, text="Classifciation", variable=problemType, value=2, style="text.TRadioButton").pack()

startSearchButton = ttk.Button(ga_frame, text="Start Search", command=startSearchThreading)
startSearchButton.grid(row=3, column=0, padx=5, pady=10)

def selectItem(a):
    curItem = tree.focus()
    item = tree.item(curItem)
    messagebox.showinfo(message=str(item))

best_results_frame = tk.LabelFrame(window, text="Search Results")
best_results_frame.pack(fill='x', padx=10, pady=10)

sub_frame = tk.Frame(best_results_frame)
sub_frame.pack(padx=10, pady=10)
    
tree = ttk.Treeview(sub_frame, show="headings")
tree["columns"] = ("1", "2", "3", "4")
    
tree.column("1", width = 80, anchor ='w')
tree.heading("1", text ="Generation")

tree.column("2", width = 100, anchor ='w')
tree.heading("2", text ="Fitness Score")

tree.column("3", width = 300, anchor ='w')
tree.heading("3", text ="Model Architecture")

tree.column("4", width = 300, anchor ='w')
tree.heading("4", text ="Model History")

tree_scroll_y = ttk.Scrollbar(sub_frame, orient="vertical", command=tree.yview)
tree_scroll_y.pack(side="right", fill="y")
tree_scroll_x = ttk.Scrollbar(sub_frame, orient="horizontal", command=tree.xview)
tree_scroll_x.pack(side="bottom", fill="x")
    
tree.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
tree.pack()
#tree.bind('<ButtonRelease-1>', selectItem)

level = tk.IntVar()
progressbar = ttk.Progressbar(orient=tk.HORIZONTAL, length=780, variable=level).pack(anchor='w', padx=20, pady=10)

ttk.Label(window, text="Developed by Atakan Şentürk").pack(anchor='w', padx=10, pady=10)

window.mainloop()