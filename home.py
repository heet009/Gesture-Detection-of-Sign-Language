import cv2
import mediapipe as mp
import numpy as np
import pickle
import joblib
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import *
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

with open('Final Project\model_asl_alphabets_paper.p', 'rb') as file:
    try:
        model_dict1 = joblib.load(file)
    except ValueError:
        file.seek(0)
        model_dict1 = pickle.load(file)

model1 = model_dict1['model']

with open('Final Project\model_isl_digits_paper.p', 'rb') as file:
    try:
        model_dict2 = joblib.load(file)
    except ValueError:
        file.seek(0)
        model_dict2 = pickle.load(file)

model2 = model_dict2['model']

with open('Final Project\model_asl_digits.p', 'rb') as file:
    try:
        model_dict3 = joblib.load(file)
    except ValueError:
        file.seek(0)
        model_dict3 = pickle.load(file)

model3 = model_dict3['model']

with open('Final Project\model.p', 'rb') as file:
    try:
        model_dict4 = joblib.load(file)
    except ValueError:
        file.seek(0)
        model_dict4 = pickle.load(file)

model4 = model_dict4['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {
    '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9',
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 
    'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q',
    'r': 'R', 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
    'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q',
    'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
}

root = ThemedTk(theme="")
root.title("Hand Gesture Recognition")
root.geometry("680x480")

style = ttk.Style()
style.theme_use("clam")

quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack(side=tk.BOTTOM, pady=10, padx=10, anchor=tk.NE)

notebook = ttk.Notebook(root,style='TNotebook',padding=6)
notebook.pack(expand=True, fill='both')

live_feed_tab = ttk.Frame(notebook,style='TFrame',height=500,width=500)
upload_image_tab = ttk.Frame(notebook,style='TFrame',height=500,width=500)

live_feed_tab.columnconfigure(0, weight=1)
live_feed_tab.columnconfigure(1, weight=1)
live_feed_tab.rowconfigure(0, weight=1)
live_feed_tab.rowconfigure(1, weight=1)
live_feed_tab.rowconfigure(2, weight=1)
live_feed_tab.rowconfigure(3, weight=1)
live_feed_tab.rowconfigure(4, weight=1)
live_feed_tab.rowconfigure(5, weight=1)

upload_image_tab.columnconfigure(0, weight=1)
upload_image_tab.columnconfigure(1, weight=1)
upload_image_tab.columnconfigure(2, weight=1)
upload_image_tab.columnconfigure(3, weight=1)
upload_image_tab.columnconfigure(4, weight=1)
upload_image_tab.columnconfigure(5, weight=1)
upload_image_tab.rowconfigure(0, weight=1)
upload_image_tab.rowconfigure(1, weight=1)
upload_image_tab.rowconfigure(2, weight=1)
upload_image_tab.rowconfigure(3, weight=1)


image = PhotoImage(file="Final Project\img.png")

notebook.add(live_feed_tab, text='Live Feed')
notebook.add(upload_image_tab, text='Upload Image')

uploaded_image_label = tk.Label(upload_image_tab)
uploaded_image_label.grid(row=0,column=0,columnspan=5,pady=10, padx=10)

pred_upload = tk.Label(upload_image_tab,text="Predicted value: ",font=("Courier", 20),bg='#dcdad5')
pred_upload.grid(row=2,column=0,columnspan=5,pady=10, padx=10)

def predict_letter(image, model, frame):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    predicted_letter = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_letter = labels_dict[prediction[0]]
            print(f"Predicted letter: {predicted_letter}")

            # Display predicted letter on the frame
            cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #cv2.imshow('frame', frame)
    return frame, predicted_letter


def predict_letter1(image, model, frame):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_letter = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            """ mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()) """

            data_aux = []
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_letter = labels_dict[prediction[0]]
            print(f"Predicted letter: {predicted_letter}")

            #cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, predicted_letter

uploaded_image_label.configure(image=image)

def predict_from_image(file_path, label,model):
    if file_path:
        frame = cv2.imread(file_path)
        result_image, predicted_letter = predict_letter1(frame.copy(), model,frame)
        
        pred_upload.config(text=f"Predicted value: {predicted_letter}")

        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        #result_image = cv2.resize(result_image, (400, 400))
        result_image = Image.fromarray(result_image)
        result_image = ImageTk.PhotoImage(image=result_image)

        label.img = result_image
        label.config(image=result_image)

def upload_image1():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")))
    predict_from_image(file_path, uploaded_image_label,model1)

def upload_image2():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")))
    predict_from_image(file_path, uploaded_image_label,model2)

def upload_image3():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")))
    predict_from_image(file_path, uploaded_image_label,model3)

def upload_image4():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*.*")))
    predict_from_image(file_path, uploaded_image_label,model4)


def display_live_feed1():

    new_window1 = tk.Toplevel(root)
    new_window1.title("Model 1")

    frame_container1 = tk.Frame(new_window1)
    frame_container1.pack()

    label1 = tk.Label(frame_container1)
    label1.pack()

    label1_pred = tk.Label(frame_container1, text="Predicted value: ",font=("Courier", 20))
    label1_pred.pack()

    def stop_feed1():
        new_window1.destroy()

    stop_button1 = tk.Button(frame_container1, text="Stop", command=stop_feed1)
    stop_button1.pack(side=tk.BOTTOM, pady=10)

    def display_feed1():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result_frame, predicted_letter = predict_letter(frame.copy(), model1,frame)
            img = Image.fromarray(result_frame)
            img = ImageTk.PhotoImage(image=img) 

            label1_pred.config(text=f"Predicted value: {predicted_letter}")

            if(hasattr(label1, 'img')):
                label1.img = img
                label1.config(image=img)
            else:
                label1.config(image=img)
                label1.img = img

            new_window1.after(10, display_feed1)  # Call this function after a delay for live feed update
    display_feed1()

def display_live_feed2():

    new_window2 = tk.Toplevel(root)
    new_window2.title("Model 2")

    frame_container2 = tk.Frame(new_window2)
    frame_container2.pack()

    label2 = tk.Label(frame_container2)
    label2.pack()

    label2_pred = tk.Label(frame_container2, text="Predicted value: ",font=("Courier", 20))
    label2_pred.pack()

    def stop_feed2():
        new_window2.destroy()

    stop_button2 = tk.Button(frame_container2, text="Stop", command=stop_feed2)
    stop_button2.pack(side=tk.BOTTOM, pady=10)

    def display_feed2():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result_frame, predicted_letter = predict_letter(frame.copy(), model2,frame)

            img = Image.fromarray(result_frame)
            img = ImageTk.PhotoImage(image=img)

            label2_pred.config(text=f"Predicted value: {predicted_letter}")

            if(hasattr(label2, 'img')):
                label2.img = img
                label2.config(image=img)
            else:
                label2.config(image=img)
                label2.img = img

            new_window2.after(10, display_feed2) 
    display_feed2()

def display_live_feed3():

    new_window3 = tk.Toplevel(root)
    new_window3.title("Model 3")

    frame_container3 = tk.Frame(new_window3)
    frame_container3.pack()

    label3 = tk.Label(frame_container3)
    label3.pack()

    label3_pred = tk.Label(frame_container3, text="Predicted value: ",font=("Courier", 20))
    label3_pred.pack()

    def stop_feed3():
        new_window3.destroy()

    stop_button3 = tk.Button(frame_container3, text="Stop", command=stop_feed3)
    stop_button3.pack(side=tk.BOTTOM, pady=10)

    def display_feed3():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result_frame, predicted_letter = predict_letter(frame.copy(), model3,frame)

            img = Image.fromarray(result_frame)
            img = ImageTk.PhotoImage(image=img)

            label3_pred.config(text=f"Predicted value: {predicted_letter}")

            if(hasattr(label3, 'img')):
                label3.img = img
                label3.config(image=img)
            else:
                label3.config(image=img)
                label3.img = img

            new_window3.after(10, display_feed3) 
    display_feed3()

def display_live_feed4():
    
        new_window4 = tk.Toplevel(root)
        new_window4.title("Model 4")
    
        frame_container4 = tk.Frame(new_window4)
        frame_container4.pack()
    
        label4 = tk.Label(frame_container4)
        label4.pack()
    
        label4_pred = tk.Label(frame_container4, text="Predicted value: ",font=("Courier", 20))
        label4_pred.pack()
    
        def stop_feed4():
            new_window4.destroy()
    
        stop_button4 = tk.Button(frame_container4, text="Stop", command=stop_feed4)
        stop_button4.pack(side=tk.BOTTOM, pady=10)
    
        def display_feed4():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                result_frame, predicted_letter = predict_letter(frame.copy(), model4,frame)
    
                img = Image.fromarray(result_frame)
                img = ImageTk.PhotoImage(image=img)
    
                label4_pred.config(text=f"Predicted value: {predicted_letter}")
    
                if(hasattr(label4, 'img')):
                    label4.img = img
                    label4.config(image=img)
                else:
                    label4.config(image=img)
                    label4.img = img
    
                new_window4.after(10, display_feed4) 
        display_feed4()

global cap
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
else:
    label_write = tk.Label(live_feed_tab, text="Select a model to use:",font=("Times New Roman", 15,"bold"),bg='#dcdad5')
    label_write.grid(row=0,column=0,pady=10, padx=10,columnspan=2)

    label_write1 = tk.Label(live_feed_tab, text="Online Datasets",font=("Courier", 10,"bold"),bg='#dcdad5')
    label_write1.grid(row=1,pady=10, padx=10,columnspan=2)

    use_model1 = tk.Button(live_feed_tab, text="ASL Alphabet", command=display_live_feed1,height=5,width=40)
    use_model1.grid(row=2,column=0,pady=10, padx=10)
   
    use_model2 = tk.Button(live_feed_tab, text="ISL Digits", command=display_live_feed2,height=5,width=40)
    use_model2.grid(row=2,column=1,pady=10, padx=10)

    use_model3 = tk.Button(live_feed_tab, text="ASL Digits", command=display_live_feed3,height=5,width=40)
    use_model3.grid(row=3,column=0, padx=10,columnspan=2)

    label_write2 = tk.Label(live_feed_tab, text="Self-Made Dataset",font=("Courier", 10,"bold"),bg='#dcdad5')
    label_write2.grid(row=4,pady=10, padx=10,columnspan=2)

    use_model4 = tk.Button(live_feed_tab, text="ASL Alphabets", command=display_live_feed4,height=5,width=40)
    use_model4.grid(row=5, padx=10,columnspan=2)


upload_label = tk.Label(upload_image_tab, text="Upload an image to predict:",font=("Times New Roman", 15,"bold"),bg='#dcdad5')
upload_label.grid(column=0,row=1,columnspan=5, pady=10)

upload_button1 = tk.Button(upload_image_tab, text="Use ASL Alphabet", command=upload_image1)
upload_button1.grid(column=1,row=3, pady=10)

upload_button2 = tk.Button(upload_image_tab, text="Use ISL Digits", command=upload_image2)
upload_button2.grid(column=2,row=3, pady=10)

upload_button3 = tk.Button(upload_image_tab, text="Use ASL Digits", command=upload_image3)
upload_button3.grid(column=3,row=3, pady=10)

upload_button4 = tk.Button(upload_image_tab, text="Use ASL Alphabets (self made dataset)", command=upload_image4)
upload_button4.grid(column=4,row=3, pady=10)

root.mainloop()
