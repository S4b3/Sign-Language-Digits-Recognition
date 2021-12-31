# Sign Language Digits Recognition

![FDS-Sapienza](https://img.shields.io/badge/FDS-Sapienza-red?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

This project was done as the final project for the **Fundamentals Of Data Science And Lab** class of Sapienza's **Data Science's Master Degree**.

## Team

- Valentino Sacco: [S4b3](https://github.com/S4b3)
- Arturo Ghinassi: [ghinassi1863151](https://github.com/ghinassi1863151)
- Camilla Savarese: [Camillasavarese](https://github.com/Camillasavarese)
- Giorgia Fontana: [GiorgiaFontana](https://github.com/GiorgiaFontana)
- Luca Romani: [LucaRomani98](https://github.com/LucaRomani98)

## Goal

For this project we tried to understand how to best perform recognition on the [Sign Language Digits Dataset](https://www.kaggle.com/ardamavi/sign-language-digits-dataset)*, starting from a Naive Bayes Classifier approach and then seeing how to enhance its performances.

## Report

An more in depth Report of our work process, observations and results can be found [here](https://github.com/S4b3/Sign-Language-Digits-Recognition/blob/main/report.pdf)

## Techniques ([main.ipynb](https://github.com/S4b3/Sign-Language-Digits-Recognition/blob/main/main.ipynb))


### The sequent approaches were tried in combination with dataset transformation techniques to find the best solution

### Naive Bayes

- Multinomial Approach
  - Preprocessing of the dataset into black and white pixels as features
    <br/>
    <br/>
    <img src = "https://user-images.githubusercontent.com/50860347/147411782-d7990395-0b38-4b60-a41d-26a246670ffa.png" style="height: 120px"/>

  - Pillow Edge Recognition  
    <br/>
    <img src = "https://user-images.githubusercontent.com/50860347/147411859-b1ff0124-37f4-41d1-9298-ea3c7907f58a.png"/><img src = "https://user-images.githubusercontent.com/50860347/147411889-56d75356-622d-4a32-812a-43e67c4b723a.png"/><img src = "https://user-images.githubusercontent.com/50860347/147411909-bf9c7a29-03da-41ce-a8b9-1c73dee337cf.png"/><img src="https://user-images.githubusercontent.com/50860347/147411931-32e7d794-12c0-4e87-9745-7ae7521bd45a.png"/><img src="https://user-images.githubusercontent.com/50860347/147411946-9e41da60-67dd-4678-aa5d-06f63c6a5f80.png"/><img src="https://user-images.githubusercontent.com/50860347/147411957-55e300c8-c3e6-4383-b5eb-9ce9cc4c46c9.png"/>
  
- Gaussian Approach
  - Standard Dataset

  - Principal Component Analysis for noise reduction

### Chamfer Distance

*to implement the chamfer distance we took this [implementation](https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367) as reference*  
We computed chamfer distance after applying above edge recognition on the dataset. This is worth mentioning but the results have not lived to our expectations.

### Support Vector Machine

- Standard Dataset
- Principal Component Analysis for noise reduction
- Gaussian Smoothing for noise reduction

    <img src="https://user-images.githubusercontent.com/50860347/147412474-ae848225-3045-47a3-b145-8146ff2c9b9c.png" style="height: 200px"/>

## Repository content


- ``main.ipynb``: is the main working area, containing all the
- ``utils.py``: module containing data loading and preprocessing functions

- ``naive_bayes_custom.py``: module containg a custom implementation of the Naive Bayes Classifier used for error understanding

- ``X.npy``: dataset images in the form of a numpy matrix
- ``Y.npy``: numpy vector containing true labels of the images

<br />
<br />
<br />
<p align="center">
    <img src="https://user-images.githubusercontent.com/50860347/147412786-183da6b0-990f-4016-9f2e-0719d8066f5b.png" style="width: 100%"/>
<p>

<br />
*"Mavi, A., (2020), “A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits”, arXiv:2011.08927 [cs.CV]"*
