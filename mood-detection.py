# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fer import FER #version 22.5.1
import cv2 ##version 4.8.1
import matplotlib.pyplot as plt ##3.7.2

emotion_detector = FER(mtcnn=True)
img = cv2.imread("4.jpg") 
analysis = emotion_detector.detect_emotions(img)
emotions_dictionary = analysis[0]['emotions'] #get the emotions key-value pair

emotion_labels  = list(emotions_dictionary.keys())

#emotion_values = emotions_dictionary.values() OR

emotion_values = [emotions_dictionary[i] for i in emotion_labels]

max_value = max(emotion_values)
for key, value in emotions_dictionary.items():
    if value == max_value:
        dominant_emotion = key.title() #make the first letter capital
        
# Define colors for each label
colors = ['blue', 'green', 'orange', 'red']

emotion_labels  = [i.title() for i in emotions_dictionary.keys()]
# Plotting the bar graph
plt.bar(emotion_labels, emotion_values, color=colors)

# Annotate the dominant emotion
index_to_annotate = emotion_labels.index(dominant_emotion)  # Index of the emotion_label to annotate
value_to_annotate = emotion_values[index_to_annotate]

# Add the annotation
plt.annotate(f'{dominant_emotion,value_to_annotate}', 
             xy=(index_to_annotate, value_to_annotate),
             xytext=(index_to_annotate, value_to_annotate),  # Adjust the text position
             ha='center',  # Horizontal alignment
             va='bottom',  # Vertical alignment
             color='black',  # Text color
             weight='bold')  # Text weight



# Adding labels and title
plt.xlabel('Emotions')
plt.ylabel('Values')
plt.title('Human moood/emotion detection')

# Display the plot
plt.show()