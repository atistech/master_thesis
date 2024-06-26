import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from src.nn_search_engine import NNSearchEngine
import threading
from tkinter import filedialog as fd

def combobox(root, text_variable, values, row, column):
    combobox = ttk.Combobox(root, textvariable=text_variable, values=values, state="readonly", width=10)
    combobox.set(values[0])
    combobox.grid(row=row, column=column, padx=5, pady=0)

def startSearchThreading():
    if(datasetSelection_value.get() == ""):
        messagebox.showwarning(title="Uyarı", message="Veriseti için bir CSV dosyası seçmelisiniz!")
    else:
        threading.Thread(target=startSearch).start()

def startSearch():
    param = {
        "dataset": datasetSelection_value.get(),
        "populationSize": populationSize_value.get(),
        "maxGenerationCount": maxGenerationCount_value.get(),
        "taskType": taskType.get()
    }
    answer = messagebox.askokcancel(message="Aramayı başlatmak istediğinize emin misiniz?")

    if answer:
        startSearchButton.config(state="disabled")
        saveButton.config(state="disabled")
        tree.delete(*tree.get_children())
        bestResultTree.delete(*bestResultTree.get_children())
        progressbar.start()
        
        search_engine = NNSearchEngine(param)
        
        global lastBestModel
        lastBestModel = None
        lastGenerationCount = 0
        for iteration_models in search_engine:
            for model in iteration_models:
                model = model.toDict()
                values_to_insert = [search_engine.generationCount, model["fitnessScore"], model["architecture"], model["history"]]
                tree.insert("", 'end', values=values_to_insert)
            bestModel = search_engine.finalBestFoundModel()
            if (lastBestModel is None):
                lastBestModel = bestModel
                lastGenerationCount = search_engine.generationCount
            else:
                if(param["taskType"]==1):
                    if(bestModel.fitnessScore < lastBestModel.fitnessScore):
                        lastBestModel = bestModel
                        lastGenerationCount = search_engine.generationCount
                else:
                    if(bestModel.fitnessScore > lastBestModel.fitnessScore):
                        lastBestModel = bestModel
                        lastGenerationCount = search_engine.generationCount
                
        bestModel_dict = lastBestModel.toDict()
        values_to_insert = [lastGenerationCount, bestModel_dict["fitnessScore"], bestModel_dict["architecture"], bestModel_dict["history"]]
        bestResultTree.insert("", 'end', values=values_to_insert)
        progressbar.stop()
        messagebox.showinfo(title="Bilgi", message="Model araması tamamlandı.")
        startSearchButton.config(state="active")
        saveButton.config(state="active")

window = tk.Tk()
window.resizable(False, False)
window.geometry("850x700")
window.title("Yapay Sinir Ağı Modeli Arama Motoru")

ttk.Label(window, text="Yapay Sinir Ağı Modeli Arama Motoru", font=("TkDefaultFont", 12)).pack(anchor="center", pady=10)

paramsFrame = tk.Label(window)
paramsFrame.pack(fill='x', padx=10, pady=10)

taskTypeFrame = tk.LabelFrame(paramsFrame, text="Görev Tipi", padx=20, pady=10)
taskTypeFrame.pack(side="left", fill="y")
taskType = tk.IntVar()
taskType.set(1)
ttk.Radiobutton(taskTypeFrame, text="Regresyon", width=21, variable=taskType, value=1).pack()
ttk.Radiobutton(taskTypeFrame, text="İkili Sınıflandırma", width=21, variable=taskType, value=2).pack()
ttk.Radiobutton(taskTypeFrame, text="Çoklu Sınıflandırma", width=21, variable=taskType, value=3).pack()

datasetSelectionFrame = tk.LabelFrame(paramsFrame, text="Veriseti Dosyası Seçimi", padx=20, pady=15)
datasetSelectionFrame.pack(side="left", fill="y", padx=5)
datasetSelection_value = tk.StringVar()
selectedFileName = tk.StringVar()
selectedFileName.set("Dosya seçilmedi.")
def select_file():
    filename = fd.askopenfilename(
        title="CSV Dosyası Seç",
        filetypes=(
            ("CSV files", "*.csv"),
        )
    )
    datasetSelection_value.set(filename)
    selectedFileName.set(filename.split("/")[-1])
datasetSelectButton = ttk.Button(datasetSelectionFrame, text="CSV Dosyası Seç", command=select_file)
datasetSelectButton.pack()
ttk.Label(datasetSelectionFrame, textvariable=selectedFileName, anchor="center", width=20).pack()

geneticParamsFrame = tk.LabelFrame(paramsFrame, text="Genetik Algoritma Parametreleri", padx=20, pady=15)
geneticParamsFrame.pack(side="left", fill="y")
populationSize_value = tk.StringVar()
ttk.Label(geneticParamsFrame, text="Populasyondaki Birey Sayısı:", width=25).grid(row=0, column=0)
combobox(geneticParamsFrame, populationSize_value, [*range(4,11,2)], row=0, column=1)
maxGenerationCount_value = tk.StringVar()
ttk.Label(geneticParamsFrame, text="Maksimum Nesil Sayısı:", width=25).grid(row=1, column=0)
combobox(geneticParamsFrame, maxGenerationCount_value, [*range(1,11)], row=1, column=1)


buttonsFrame = tk.LabelFrame(paramsFrame, text="Operasyonlar", padx=10, pady=10)
buttonsFrame.pack(side="left", fill="y", padx=5)
startSearchButton = ttk.Button(buttonsFrame, text="BAŞLAT", command=startSearchThreading)
startSearchButton.grid(row=0, column=0, pady=5)
progressbar = ttk.Progressbar(buttonsFrame, orient=tk.HORIZONTAL, mode="indeterminate")
progressbar.grid(row=1, column=0)

def selectItem(a):
    curItem = tree.focus()
    item = tree.item(curItem)
    messagebox.showinfo(message=str(item))

searchResultsframe = tk.LabelFrame(window, text="Arama Sonuçları", padx=15, pady=15)
searchResultsframe.pack(fill=tk.BOTH, expand=True, padx=10)
    
tree = ttk.Treeview(searchResultsframe, columns=("1", "2", "3", "4"), show="headings")
    
tree.column("1", width = 50, anchor ='c')
tree.heading("1", text ="Nesil")

tree.column("2", width = 100, anchor ='c')
tree.heading("2", text ="Fitness Skoru")

tree.column("3", width = 350, anchor ='c')
tree.heading("3", text ="Model Mimarisi")

tree.column("4", width = 220, anchor ='c')
tree.heading("4", text ="Model Metrik Sonuçları")

tree_scroll_y = ttk.Scrollbar(searchResultsframe, orient="vertical", command=tree.yview)
tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
tree.configure(yscrollcommand=tree_scroll_y.set)
tree.pack(fill=tk.BOTH, expand=True)
#tree.bind('<ButtonRelease-1>', selectItem)

bestResultframe = tk.LabelFrame(window, text="En İyi Arama Sonucu", padx=20, pady=10)
bestResultframe.pack(fill='x', padx=10)

bestResultTree = ttk.Treeview(bestResultframe, columns=("1", "2", "3", "4"), show="headings", height=1)
bestResultTree.column("1", width = 50, anchor ='c')
bestResultTree.heading("1", text ="Nesil")
bestResultTree.column("2", width = 100, anchor ='c')
bestResultTree.heading("2", text ="Fitness Skoru")
bestResultTree.column("3", width = 350, anchor ='c')
bestResultTree.heading("3", text ="Model Mimarisi")
bestResultTree.column("4", width = 220, anchor ='c')
bestResultTree.heading("4", text ="Model Metrik Sonuçları")
bestResultTree.pack(fill=tk.BOTH, expand=True)

def saveModel():
    saveButton.config(state="disabled")
    folderName = fd.askdirectory(
        title="Klasör Seç"
    )
    lastBestModel.model.save(f"{folderName}/bestModel.{fileType.get()}")
    messagebox.showinfo(title="Bilgi", message="Model kaydedildi.")
    saveButton.config(state="active")

fileTypeFrame = tk.LabelFrame(bestResultframe, text="Model Kaydetme Ayarları", padx=20, pady=10)
fileTypeFrame.pack()
fileType = tk.StringVar()
fileType.set("h5")
ttk.Radiobutton(fileTypeFrame, text=".h5", width=21, variable=fileType, value="h5").pack(side="left")
ttk.Radiobutton(fileTypeFrame, text=".keras", width=21, variable=fileType, value="keras").pack(side="left")
saveButton = ttk.Button(fileTypeFrame, text="Modeli Kaydet", command=saveModel)
saveButton.pack(side="left")
saveButton.config(state="disabled")

ttk.Label(window, text="Developed by Atakan Şentürk").pack(anchor='w', padx=10, pady=10)

window.mainloop()