import tkinter as tk
import main
from tkinter import colorchooser
class StartMenu():
    def __init__(self):
        self.root = tk.Tk()
        self.createMenu()
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.maskColor = (255, 0, 255)
        self.root.mainloop()
    def createMenu(self):
        self.titleLabel = tk.Label(self.root, text="3D \nOBJECT LOCALISATION \nUSING RGB-D ", font = "Arial 40")
        self.titleLabel.pack(side = "top")
        self.eyeMovement1 = tk.PhotoImage(file = "eye1.png")
        self.eyeState = 1
        self.startButton = tk.Button(self.root, image = self.eyeMovement1, command=lambda: self.start(self.startButton), bd = 0)
        self.settingsButton = tk.Button(self.root, text = "Settings", font = "Arial 25", bd = 0, command=self.openSettingsMenu)
        self.settingsButton.pack(side = "bottom")
        self.startButton.pack(side = "bottom")
    def start(self, eyeButton, ):
        if self.eyeState == 0:
            self.eyeMovement1 = tk.PhotoImage(file = "eye1.png")
            eyeButton.configure(image = self.eyeMovement1)
            self.eyeState = 1
        else:
            self.eyeMovement2 = tk.PhotoImage(file = "eye2.png")
            eyeButton.configure(image = self.eyeMovement2)
            self.eyeState = 0
        main.main(self.maskColor)
    def openSettingsMenu(self):
        self.settingsMenu = SettingsMenu(self)
        print(self.maskColor)
class SettingsMenu():
    def __init__(self, startMenu):
        self.root = tk.Tk()
        self.startMenu = startMenu
        self.root.title("Settings Menu")
        self.root.geometry("640x480")
        #self.protocol("WM_DELETE_WINDOW", self.onClosing)
        tk.Label(self.root, text="Settings", font=("Arial", 24)).pack(pady=10)
        
        main = tk.Frame(self.root)
        main.pack(expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, padx=20)
        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, padx=20)

        # Left Frame
        tk.Label(left, text="Label Color:").pack(pady=5)
        label_color_frame = tk.Frame(left)
        label_color_frame.pack()
        self.label_color_button = tk.Button(label_color_frame, text="Select Label Color", command=lambda: self.choose_color("label"))
        self.label_color_button.pack(side=tk.LEFT)
        self.label_color_display = tk.Label(label_color_frame, width=3, height=1, bg="white", relief=tk.SUNKEN)
        self.label_color_display.pack(side=tk.LEFT, padx=5)

        tk.Label(left, text="Box Color:").pack(pady=5)
        box_color_frame = tk.Frame(left)
        box_color_frame.pack()
        self.box_color_button = tk.Button(box_color_frame, text="Select Box Color", command=lambda: self.choose_color("box"))
        self.box_color_button.pack(side=tk.LEFT)
        self.box_color_display = tk.Label(box_color_frame, width=3, height=1, bg="white", relief=tk.SUNKEN)
        self.box_color_display.pack(side=tk.LEFT, padx=5)

        tk.Label(left, text="Mask Color:").pack(pady=5)
        mask_color_frame = tk.Frame(left)
        mask_color_frame.pack()
        self.mask_color_button = tk.Button(mask_color_frame, text="Select Mask Color", command=lambda: self.chooseMask("mask"))
        self.mask_color_button.pack(side=tk.LEFT)
        self.mask_color_display = tk.Label(mask_color_frame, width=3, height=1, bg="white", relief=tk.SUNKEN)
        self.mask_color_display.pack(side=tk.LEFT, padx=5)

        tk.Label(left, text="Alpha:").pack(pady=5)
        self.alpha = tk.DoubleVar(value=0.5)
        tk.Scale(left, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.alpha).pack()

        # Right Frame
        self.padding = tk.IntVar(value=10)
        self.confidence = tk.DoubleVar(value=0.5)

        tk.Label(right, text="Padding:").pack(pady=5)
        padding_frame = tk.Frame(right)
        padding_frame.pack()
        self.padding_entry = tk.Entry(padding_frame, textvariable=self.padding, width=5)
        self.padding_entry.pack(side=tk.LEFT)
        tk.Button(padding_frame, text="-", command=lambda: self.change_value(self.padding, -1)).pack(side=tk.LEFT)
        tk.Button(padding_frame, text="+", command=lambda: self.change_value(self.padding, 1)).pack(side=tk.LEFT)

        tk.Label(right, text="Confidence:").pack(pady=5)
        confidence_frame = tk.Frame(right)
        confidence_frame.pack()
        self.confidence_entry = tk.Entry(confidence_frame, textvariable=self.confidence, width=5)
        self.confidence_entry.pack(side=tk.LEFT)
        tk.Button(confidence_frame, text="-", command=lambda: self.change_value(self.confidence, -0.01)).pack(side=tk.LEFT)
        tk.Button(confidence_frame, text="+", command=lambda: self.change_value(self.confidence, 0.01)).pack(side=tk.LEFT)

        tk.Label(right, text="Model Weights:").pack(pady=10)
        tk.Button(right, text="Select YOLO Weights", command=self.select_yolo).pack(pady=5)
        tk.Button(right, text="Select SAM Weights", command=self.select_sam).pack(pady=5)

        tk.Label(right, text="Source:").pack(pady=10)
        self.source = tk.StringVar(value="webcam")
        self.webcam_radio = tk.Radiobutton(right, text="Webcam", variable=self.source, value="webcam", command=self.toggle_file)
        self.webcam_radio.pack()
        self.dataset_radio = tk.Radiobutton(right, text="Dataset", variable=self.source, value="dataset", command=self.toggle_file)
        self.dataset_radio.pack()

        self.file_btn = tk.Button(right, text="Select File", command=self.select_file)
        self.file_lbl = tk.Label(right, text="No file selected")

    def choose_color(self, attr):
        self.startMenu.maskColor = colorchooser.askcolor(title ="Choose color")
    def change_value(self, var, amount):
        var.set(round(var.get() + amount, 2))

    def select_yolo(self):
        tk.filedialog.askopenfilename()

    def select_sam(self):
        tk.filedialog.askopenfilename()

    def toggle_file(self):
        if self.source.get() == "dataset":
            self.file_btn.pack(pady=5)
            self.file_lbl.pack(pady=5)
        else:
            self.file_btn.pack_forget()
            self.file_lbl.pack_forget()

    def select_file(self):
        f = tk.filedialog.askopenfilename()
        if f:
            self.file_lbl.config(text=f)
StartMenu()