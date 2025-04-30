import tkinter as tk
import main_gui
from tkinter import colorchooser, filedialog
class StartMenu():
    settingsMenu = None
    label_color = (255, 0, 0)
    box_color = (255, 0, 0)
    mask_color = (255, 0, 255)
    alpha = 0.4
    pad = 30
    confidence_threeshold = 0.5
    yolo_weights_filename = "yolo-Weights/yolov8n.pt"
    sam_weights_filename = "fastsam-weights/FastSAM-x.pt"
    dataset_path = "scene_02"
    use_data_set = True
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.createMenu()
        self.root.mainloop()
        
    def createMenu(self):
        self.titleLabel = tk.Label(self.root, text="3D \nOBJECT LOCALIZATION \nUSING RGB-D ", font = "Arial 40")
        self.titleLabel.pack(side = "top")
        self.eyeMovement1 = tk.PhotoImage(file = "eye1.png")
        self.eyeState = 1
        self.startButton = tk.Button(self.root, image = self.eyeMovement1, command=lambda: self.start(self.startButton), bd = 0)
        self.settingsButton = tk.Button(self.root, text = "Settings", font = "Arial 25", bd = 0, command=self.openSettingsMenu)
        self.settingsButton.pack(side = "bottom")
        self.startButton.pack(side = "bottom")
    def start(self, eyeButton):
        if self.eyeState == 0:
            self.eyeMovement1 = tk.PhotoImage(file = "eye1.png")
            eyeButton.configure(image = self.eyeMovement1)
            self.eyeState = 1
        else:
            self.eyeMovement2 = tk.PhotoImage(file = "eye2.png")
            eyeButton.configure(image = self.eyeMovement2)
            self.eyeState = 0
        self.video = main_gui.Main(__class__.label_color, 
                                   __class__.box_color, 
                                   __class__.mask_color, 
                                   __class__.alpha, 
                                   __class__.pad, 
                                   __class__.confidence_threeshold,
                                   __class__.sam_weights_filename,
                                   __class__.yolo_weights_filename,
                                   __class__.dataset_path,
                                   __class__.use_data_set).main()
        
    def closeSettingsMenu():
        __class__.settingsMenu = None
    def openSettingsMenu(self):
        if __class__.settingsMenu == None:
            __class__.settingsMenu = SettingsMenu(self.root)
        else:
            return

class SettingsMenu():
    def __init__(self, startMenu):
        self.root = tk.Toplevel(startMenu)
        self.startMenu = startMenu
        self.root.title("Settings Menu")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.onClosing)
        tk.Label(self.root, text="Settings", font=("Arial", 24)).pack(pady=10)
        
        main = tk.Frame(self.root)
        main.pack(expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, padx=20)
        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, padx=20)

        # Left Frame
        #Code for choosing Label Color 
        tk.Label(left, text="Label Color:").pack(pady=5)
        label_color_frame = tk.Frame(left)
        label_color_frame.pack()
        self.label_color_button = tk.Button(label_color_frame, text="Select Label Color", command=lambda: self.choose_color("label"))
        self.label_color_button.pack(side=tk.LEFT)
        self.label_color_display = tk.Label(label_color_frame, width=3, height=1, bg= self.createColorForTkinter(StartMenu.label_color), relief=tk.SUNKEN)
        self.label_color_display.pack(side=tk.LEFT, padx=5)

        #Code for choosing Box Color
        tk.Label(left, text="Box Color:").pack(pady=5)
        box_color_frame = tk.Frame(left)
        box_color_frame.pack()
        self.box_color_button = tk.Button(box_color_frame, text="Select Box Color", command=lambda: self.choose_color("box"))
        self.box_color_button.pack(side=tk.LEFT)
        self.box_color_display = tk.Label(box_color_frame, width=3, height=1, bg= self.createColorForTkinter(StartMenu.box_color), relief=tk.SUNKEN)
        self.box_color_display.pack(side=tk.LEFT, padx=5)

        #Code for choosing Mask Color
        tk.Label(left, text="Mask Color:").pack(pady=5)
        mask_color_frame = tk.Frame(left)
        mask_color_frame.pack()
        self.mask_color_button = tk.Button(mask_color_frame, text="Select Mask Color", command=lambda: self.choose_color("mask"))
        self.mask_color_button.pack(side=tk.LEFT)
        self.mask_color_display = tk.Label(mask_color_frame, width=3, height=1, bg= self.createColorForTkinter(StartMenu.mask_color), relief=tk.SUNKEN)
        self.mask_color_display.pack(side=tk.LEFT, padx=5)

        #Code for choosing Alpha
        tk.Label(left, text="Alpha:").pack(pady=5)
        self.alphaSlider = tk.DoubleVar(value=StartMenu.alpha)
        self.scale = tk.Scale(left, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.alphaSlider, command=self.choose_alpha)
        self.scale.pack()

        # Right Frame
        self.padding = tk.IntVar(value=10)
        self.confidence = tk.DoubleVar(value=0.5)

        #Code for changing Padding
        tk.Label(right, text="Padding:").pack(pady=5)
        padding_frame = tk.Frame(right)
        padding_frame.pack()
        self.padding_label = tk.Label(padding_frame, text = str(StartMenu.pad), width=5, bg = "white")
        self.padding_label.pack(side=tk.LEFT)
        self.padButtonMinus = tk.Button(padding_frame, text="-", command=lambda: self.change_value(StartMenu.pad, -1, self.padding_label))
        self.padButtonMinus.pack(side=tk.LEFT)
        self.padButtonPlus = tk.Button(padding_frame, text="+", command=lambda: self.change_value(StartMenu.pad, 1, self.padding_label))
        self.padButtonPlus.pack(side=tk.LEFT)

        #Code for chaning confidence
        tk.Label(right, text="Confidence Threeshold:").pack(pady=5)
        self.confidenceSlider = tk.DoubleVar(value=StartMenu.confidence_threeshold)
        self.scaleConf = tk.Scale(right, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.confidenceSlider, command=self.choose_confidence_threshold)
        self.scaleConf.pack()

        #Code for selecting Model Weights Filepath
        tk.Label(right, text="Model Weights:").pack(pady=10)
        tk.Button(right, text="Select YOLO Weights", command=self.select_yolo).pack(pady=5)
        tk.Button(right, text="Select SAM Weights", command=self.select_sam).pack(pady=5)

        #Code for selecting source
        tk.Label(right, text="Source:").pack(pady=10)
        self.source = tk.StringVar(value="dataset")
        self.webcam_radio = tk.Radiobutton(right, text="Webcam", variable=self.source, value="webcam", command=self.toggle_file)
        self.webcam_radio.pack()
        self.dataset_radio = tk.Radiobutton(right, text="Dataset", variable=self.source, value="dataset", command=self.toggle_file)
        self.dataset_radio.pack()

        self.file_btn = tk.Button(right, text="Select File", command=self.select_file)
        self.file_lbl = tk.Label(right, text="No file selected")

    def choose_color(self, attr):
        if attr == "label":
            StartMenu.label_color = colorchooser.askcolor(title ="Choose color")[0]
            print(StartMenu.label_color)
            self.label_color_display.config(bg = self.createColorForTkinter(StartMenu.label_color))
        elif attr == "box":
            StartMenu.box_color = colorchooser.askcolor(title ="Choose color")[0]
            print(StartMenu.box_color)
            self.box_color_display.config(bg = self.createColorForTkinter(StartMenu.box_color))
        elif attr == "mask":
            StartMenu.mask_color = colorchooser.askcolor(title ="Choose color")[0]
            print(StartMenu.mask_color)
            self.mask_color_display.config(bg = self.createColorForTkinter(StartMenu.mask_color))

    def choose_alpha(self, alpha):
        StartMenu.alpha = float(alpha)
    
    def change_value(self, var, amount, label):
        if(var >= 1 or (var == 0 and amount > 0)):
            var += amount
            print(var)
        StartMenu.pad = var
        label.config(text = str(var))

    def choose_confidence_threshold(self, confidenceThreshold):
        StartMenu.confidence_threeshold = float(confidenceThreshold)


    def select_yolo(self):
        self.filepath = tk.filedialog.askopenfilename()
        if  self.filepath!= None:
            StartMenu.yolo_weights_filename = self.filepath
            print(f"Filepath for YOLO selected\n{self.filepath}")

    def select_sam(self):
        self.filepath = tk.filedialog.askopenfilename()
        if  self.filepath!= None:
            StartMenu.sam_weights_filename = self.filepath
            print(f"Filepath for SAM selected\n{self.filepath}")

    def toggle_file(self):
        if self.source.get() == "dataset":
            StartMenu.use_data_set = True
            self.file_btn.pack(pady=5)
            self.file_lbl.pack(pady=5)
        else:
            self.file_btn.pack_forget()
            self.file_lbl.pack_forget()
            StartMenu.use_data_set = False

    def select_file(self):
        data_path_filename = tk.filedialog.askdirectory()
        if data_path_filename:
            self.file_lbl.config(text=data_path_filename, font = "Arial 8")
            StartMenu.dataset_path = data_path_filename

    def onClosing(self):
        StartMenu.closeSettingsMenu()
        self.root.destroy()

    def createColorForTkinter(self, tupleColor):
        color = '#'
        for c in tupleColor:
            if c == 0:
                color+='00'
            else:
                color += str(hex(c))[2:]
        return color
    def createColorForYolo(self, string):
        print(string)
        color = []
        if(string == None):
            return main_gui.Main.label_color
        for i in range(1, len(string), 2):
            color.append(int(string[i]+string[i+1], 16))
        print(reversed(color))
        return tuple(reversed(color))
        
StartMenu()