# face-mask-detection
Predict if a person is wear a mask or not.
- - - -
![Face-Blurring](https://github.com/Ahmad3bdelHafeez/face-mask-detection/blob/main/output%202.PNG "Face-Blurring")
# High level pseudo-code for input image called it ‘im’:
1.	Get faces in ‘im’ by used ‘CaffeeModel’ Pre-trained model for face detection.
2.	For each face in ‘im’:
    1.	Calculate the boundary box of the face.
    2.	Get area of the boundary box from ‘im’.
    3.	Predict if the face is with mask or not by my own trained model.

# Demo: [[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H3V2GYPotTbgHEz7c3qRtJlOhjGP0bAN?usp=sharing)]
# How to run:
1.	Install the dependencies packages from ‘requirements.txt’ file.
2.	Write your test image path in ‘main.py’ file.
3.	Run ‘main.py’ file and see the results.
# References:
1.	https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ 
2.	https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-08-02-03-Advanced-Operations-Detecting-Faces-and-Features.ipynb#scrollTo=MApLLAYcGX31

