import tkinter as tk

root = tk.Tk()

for i in range(5):  # 5 satır
    for j in range(3):  # 3 sütun
        cell = tk.Label(root, text=f"Cell {i+1}-{j+1}asdsadassasasd sdasdasdsad", borderwidth=1, relief="solid", width=10, height=3)
        cell.grid(row=i, column=j)

root.mainloop()
