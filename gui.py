import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from utilities import *
from Hopfield import HopfieldNetwork


class App:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Predictions")
        self.width = 1000
        self.height = 550
        self.window.resizable(False, False)
        self.window.geometry(str(self.width) + "x" + str(self.height))
        self.input_arr = None
        self.network = HopfieldNetwork()

        load_button = tk.Button(
            master=self.window,
            command=lambda: self.load_action(),
            height=2,
            width=10,
            text="Load"
        )
        load_button.grid(row=1, column=1)

        self.window.mainloop()

    def load_action(self):
        input_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select a file",
            filetypes=(("PNG files", ".jpg"), ("all files", "*.*"))
        )
        if not input_path:
            tk.messagebox.showerror(title="FileOpenError", message="File not Found")
            return

        self.input_arr = change_image(input_path, 5)
        self.plot(self.input_arr, 2, 1)

        predict_button = tk.Button(
            master=self.window,
            command=lambda: self.predict(self.input_arr),
            height=2,
            width=10,
            text="Predict"
        )
        predict_button.grid(row=1, column=2)

        return

    def plot(self, array, row, col):

        # the figure that will contain the plot
        fig = Figure(figsize=(5, 5),
                     dpi=100)

        # adding the subplot
        plot1 = fig.add_subplot(111)

        # plotting the graph
        plot1.imshow(array)

        # creating the Tkinter canvas containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                   master=self.window)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid(row=row, column=col)

    def predict(self, image):
        # TODO Tutaj funkcja predict() z pliku Hopfield.py
        predicted_arr = self.network.predict_image(image)
        self.plot(predicted_arr, 2, 2)


if __name__ == '__main__':
    App()
