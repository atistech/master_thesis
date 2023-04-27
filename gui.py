import tkinter as tk
from tkinter import ttk
import nn.Params as params

def combobox(root, text, values, row, column, isReadonly):
    label = ttk.Label(root, text=text)
    label.grid(row=row, column=column-1, padx=5, pady=10)
    combobox = ttk.Combobox(root, textvariable=tk.StringVar(), values=values)
    combobox.set(values[0])
    combobox.grid(row=row, column=column, padx=5, pady=10)
    if isReadonly:
        combobox.config(state="disabled")
    return combobox

def entry(root, text, value, row, column, isReadonly):
    label = ttk.Label(root, text=text)
    label.grid(row=row, column=column-1, padx=5, pady=10)
    entry = ttk.Entry(root, textvariable=value)
    entry.grid(row=row, column=column, padx=5, pady=10)
    if isReadonly:
        entry.config(state="readonly")
    return entry

def load():
    tree1.delete(*tree1.get_children())
    for i in range(50):
        tree1.insert("", 'end', values=(str(i), str(90), str(88),"dense-softmax-10", "dense-softmax-10", "dense-softmax-10", "dense-softmax-10"))
        tree2.insert("", 'end', values=(str(i), str(90), str(88),"dense-softmax-10", "dense-softmax-10", "dense-softmax-10", "dense-softmax-10"))

def deneme():
    from GeneticAlgorithm import GeneticAlgorithm
    generationCount = 0
    params_dict = {
        "datasetSelection": 1
    }
    algorithm = GeneticAlgorithm(popSize=5, param_dict=params_dict)
    algorithm.initialPopulation()
    results = algorithm.calculateFitness()
    for result in results:
        values_to_insert = [generationCount, result.accuracy, result.loss, result.val_accuracy, result.val_loss, result.toString()]
        tree2.insert("", 'end', values=values_to_insert)
    result = algorithm.populationResult()
    values_to_insert = [generationCount, result.accuracy, result.loss, result.val_accuracy, result.val_loss, result.toString()]
    tree1.insert("", 'end', values=values_to_insert)

window = tk.Tk()
window.resizable(False, False)
window.title('Automatic Neural Network Search')

output_string = tk.StringVar()
input_string = tk.StringVar()
epoch_string = tk.StringVar()
batch_size_string = tk.StringVar()

dataset_frame = tk.LabelFrame(window, text="Dataset Settings")
dataset_frame.grid(row=0, column=0, padx=10, pady=10)

problem_type_select = combobox(dataset_frame, "Problem Type:", ("Multiclass Classification", "Binary Classification"), 0, 1, True)
dataset_select = combobox(dataset_frame, "Dataset Selection:", ("Mnist", "Fashion Mnist"), 1, 1, False)
spacer = ttk.Label(dataset_frame).grid(row=1, column=2, padx=20, pady=10)
train_test_split_select = combobox(dataset_frame, "Train-Test Split Ratio:", (0.2, 0.33), 1, 4, False)

input_select = entry(dataset_frame, "Input Layer:", input_string, 2, 1, True)
input_string.set("(28*28, )")
spacer2 = ttk.Label(dataset_frame).grid(row=1, column=2, padx=20, pady=10)
output_select = entry(dataset_frame, "Output Layer:", output_string, 2, 4, True)
output_string.set("Dense Softmax 10")

model_frame = tk.LabelFrame(window, text="Model Hyperparameters Settings")
model_frame.grid(row=1, column=0, padx=10, pady=10)
loss_function_select = combobox(model_frame, "Loss Function:", params.ModelLossFunctions, 1, 1, True)
spacer = ttk.Label(model_frame).grid(row=1, column=2, padx=20, pady=10)
optimizer_algorithm_select = combobox(model_frame, "Optimizer Algorithm:", params.ModelOptimizers, 1, 4, True)

epoch_select = entry(model_frame, "Amount of Epoch:", epoch_string, 2, 1, True)
epoch_string.set("5")
spacer2 = ttk.Label(model_frame).grid(row=2, column=2, padx=20, pady=10)
batch_size_select = entry(model_frame, "Batch Size:", batch_size_string, 2, 4, True)
batch_size_string.set("600")

start_button = ttk.Button(model_frame, text="START", command=load)
start_button.grid(row=3, column=0, padx=10, pady=10)

deneme_button = ttk.Button(model_frame, text="DENEME", command=deneme)
deneme_button.grid(row=3, column=1, padx=10, pady=10)

def createResultsView(root):
    sub_frame = tk.Frame(root)
    sub_frame.pack(padx=10, pady=10)
    
    tree = ttk.Treeview(sub_frame, show="headings", selectmode="extended")
    tree["columns"] = ("1", "2", "3", "4", "5", "6", "7")
    for i in range(7):
        if i==6:
            tree.column(str(i), width = 200, anchor ='w')
        else:
            tree.column(str(i), width = 100, anchor ='c')
    tree.heading("1", text ="Generation")
    tree.heading("2", text ="Accuracy")
    tree.heading("3", text ="Loss")
    tree.heading("4", text ="Val-Accuracy")
    tree.heading("5", text ="Val-Loss")
    tree.heading("6", text ="Model Architecture")

    tree_scroll_y = tk.Scrollbar(sub_frame, command=tree.yview)
    tree_scroll_y.pack(side="right", fill="y")
    
    tree.configure(yscrollcommand=tree_scroll_y.set)
    tree.pack()
    return tree

best_results_frame = tk.LabelFrame(window, text="Best Results")
best_results_frame.grid(row=2, column=0, padx=10, pady=10)
tree1 = createResultsView(best_results_frame)

all_results_frame = tk.LabelFrame(window, text="All Results")
all_results_frame.grid(row=3, column=0, padx=10, pady=10)
tree2 = createResultsView(all_results_frame)

window.mainloop()