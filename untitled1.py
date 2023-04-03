
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
  

####import model

model = pickle.load(open('RandomForestClassifier','rb'))

def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs

def prediction(Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting):     
    prediction = classifier.predict(
        [[Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
    st.title("OPTIMIZATION OF MACHINE DOWNTIME")
      
   
    
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    Machine_ID= st.text_input("Machine_ID", "Type Here")
    Hydraulic_Pressure = st.text_input("Hydraulic_Pressure", "Type Here")
    Coolant_Pressure = st.text_input("Coolant_Pressure", "Type Here")
    Air_System_Pressure = st.text_input("Air_System_Pressure ", "Type Here")
    Coolant_Temperature = st.text_input("Coolant_Temperature", "Type Here")
    Spindle_Vibration = st.text_input("Spindle_Vibration", "Type Here")
    Spindle_Speed = st.text_input(" Spindle_Speed", "Type Here")
    Tool_Vibration = st.text_input("Tool_Vibration", "Type Here")
    Voltage = st.text_input("Voltage", "Type Here")
    Torque = st.text_input("Torque", "Type Here")
    Cutting = st.text_input("Cutting", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()
