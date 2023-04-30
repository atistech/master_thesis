import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import nn.Params as params
from GeneticAlgorithm import GeneticAlgorithm
import threading

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
        "datasetSelection": datasetSelection_value.get(),
        "trainTestSplit": trainTestSplit_value.get(),
        "lossFunction": lossFunction_value.get(),
        "optimizer": optimizer_value.get(),
        "epochs": epoch_value.get(),
        "batchSize": batchSize_value.get(),
        "populationSize": populationSize_value.get()
    }
    answer = messagebox.askokcancel(message=param)

    if answer:
        startButton.config(state="disabled")
        tree1.delete(*tree1.get_children())
        tree2.delete(*tree2.get_children())
        
        generationCount = 0
        algorithm = GeneticAlgorithm(param_dict=param)
        algorithm.initialPopulation()
        fitnessResults = algorithm.calculateFitness()
        populationResult = algorithm.populationResult()
        addResultsToTreeview(generationCount, fitnessResults, populationResult)
        
        while generationCount < int(maxGenerationCount_value.get()):
            generationCount += 1
            algorithm.selection()
            algorithm.crossOver()
            algorithm.mutation()
            fitnessResults = algorithm.calculateFitness()
            populationResult = algorithm.populationResult()
            addResultsToTreeview(generationCount, fitnessResults, populationResult)
        
        startButton.config(state="active")

window = tk.Tk()
window.resizable(False, False)
window.title("Neural Network Model Search Engine")

parameters_frame = tk.Frame(window)
parameters_frame.grid(row=0, column=0, padx=10, pady=10)
results_frame = tk.Frame(window)
results_frame.grid(row=0, column=1, padx=10, pady=10)

### Main Control Frame ###
top_frame = tk.Frame(parameters_frame)
top_frame.pack(fill='both', pady=60)
ttk.Label(top_frame, text="Neural Network Model Search Engine", font=("Arial", 14)).pack(anchor="center", pady=10)
startButton = tk.Button(top_frame, text="Start Search", font=("Arial", 10), command=startSearchThreading)
startButton.pack(padx=5, pady=5)

### Dataset Parameters Frame ###
dataset_frame = tk.LabelFrame(parameters_frame, text="Dataset Parameters")
dataset_frame.pack()
problemType_value = tk.StringVar()
ttk.Label(dataset_frame, text="Problem Type:").grid(row=0, column=0, padx=20, pady=10)
combobox(dataset_frame, problemType_value, ("Multiclass Classification", "Binary Classification"), 0, 1)
datasetSelection_value = tk.StringVar()
ttk.Label(dataset_frame, text="Dataset Selection:").grid(row=1, column=0, padx=20, pady=10)
combobox(dataset_frame, datasetSelection_value, ("Mnist", "Fashion Mnist"), 1, 1)
trainTestSplit_value = tk.StringVar()
ttk.Label(dataset_frame, text="Train-Test Split Ratio:").grid(row=1, column=3, padx=20, pady=10)
combobox(dataset_frame, trainTestSplit_value, (0.2, 0.33), 1, 4)
inputLayer_value = tk.StringVar()
inputLayer_value.set("(28*28, )")
ttk.Label(dataset_frame, text="Input Layer:").grid(row=2, column=0, padx=5, pady=10)
ttk.Entry(dataset_frame, textvariable=inputLayer_value, state="readonly").grid(row=2, column=1, padx=5, pady=10)
ttk.Label(dataset_frame).grid(row=1, column=2, padx=20, pady=10)
outputLayer_value = tk.StringVar()
outputLayer_value.set("Dense Softmax 10")
ttk.Label(dataset_frame, text="Output Layer:").grid(row=2, column=3, padx=5, pady=10)
ttk.Entry(dataset_frame, textvariable=outputLayer_value, state="readonly").grid(row=2, column=4, padx=5, pady=10)

### Neural Network Model Hyperparameters Frame ###
model_frame = tk.LabelFrame(parameters_frame, text="Neural Network Model Hyperparameters")
model_frame.pack(fill='x')
lossFunction_value = tk.StringVar()
ttk.Label(model_frame, text="Loss Function:").grid(row=1, column=0, padx=5, pady=10)
combobox(model_frame, lossFunction_value, params.ModelLossFunctions, 1, 1)
ttk.Label(model_frame).grid(row=1, column=2, padx=20, pady=10)
optimizer_value = tk.StringVar()
ttk.Label(model_frame, text="Optimizer Algorithm:").grid(row=1, column=3, padx=5, pady=10)
combobox(model_frame, optimizer_value, params.ModelOptimizers, 1, 4)
epoch_value= tk.StringVar()
epoch_value.set("5")
ttk.Label(model_frame, text="Epoch:").grid(row=2, column=0, padx=5, pady=10)
ttk.Entry(model_frame, textvariable=epoch_value, state="readonly").grid(row=2, column=1, padx=5, pady=10)
ttk.Label(model_frame).grid(row=2, column=2, padx=20, pady=10)
batchSize_value = tk.StringVar()
batchSize_value.set("600")
ttk.Label(model_frame, text="Batch Size:").grid(row=2, column=3, padx=5, pady=10)
ttk.Entry(model_frame, textvariable=batchSize_value, state="readonly").grid(row=2, column=4, padx=5, pady=10)

### Genetic Algorithm Parameters Frame ###
ga_frame = tk.LabelFrame(parameters_frame, text="Genetic Algorithm Parameters")
ga_frame.pack(fill='x')
populationSize_value = tk.StringVar()
ttk.Label(ga_frame, text="Population Size:").grid(row=0, column=0, padx=5, pady=10)
combobox = ttk.Combobox(ga_frame, textvariable=populationSize_value, values=[i for i in range(2,11)], state="readonly")
combobox.current(8)
combobox.grid(row=0, column=1, padx=5, pady=10)
ttk.Label(ga_frame).grid(row=0, column=2, padx=20, pady=10)
maxGenerationCount_value = tk.StringVar()
ttk.Label(ga_frame, text="Max Generation Count:").grid(row=0, column=3, padx=5, pady=10)
combobox = ttk.Combobox(ga_frame, textvariable=maxGenerationCount_value, values=[i for i in range(1,11)], state="readonly")
combobox.current(9)
combobox.grid(row=0, column=4, padx=5, pady=10)

ttk.Label(parameters_frame, text="Developed by Atakan Şentürk").pack(anchor='w', pady=10)

def createResultsTreeview(root):
    sub_frame = tk.Frame(root)
    sub_frame.pack(padx=10, pady=10)
    
    tree = ttk.Treeview(sub_frame, show="headings", selectmode="extended")
    tree["columns"] = ("1", "2", "3", "4", "5", "6")
    for i in range(6):
        if i==6:
            tree.column(str(i), width = 300, anchor ='w')
        else:
            tree.column(str(i), width = 80, anchor ='c')
    tree.heading("1", text ="Generation")
    tree.heading("2", text ="Accuracy")
    tree.heading("3", text ="Loss")
    tree.heading("4", text ="Val-Accuracy")
    tree.heading("5", text ="Val-Loss")
    tree.heading("6", text ="Model Architecture")

    tree_scroll_y = tk.Scrollbar(sub_frame, orient="vertical", command=tree.yview)
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = tk.Scrollbar(sub_frame, orient="horizontal", command=tree.xview)
    tree_scroll_x.pack(side="bottom", fill="x")
    
    tree.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
    tree.pack()
    return tree

best_results_frame = tk.LabelFrame(results_frame, text="Best Results")
best_results_frame.pack()
tree1 = createResultsTreeview(best_results_frame)

all_results_frame = tk.LabelFrame(results_frame, text="All Results")
all_results_frame.pack()
tree2 = createResultsTreeview(all_results_frame)

window.mainloop()