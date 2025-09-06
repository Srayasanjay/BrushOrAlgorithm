# BrushOrAlgorithm

This project explores the use of deep learning to classify artwork as either AI-generated or human-made. With the rise of generative models producing unlabeled artwork, the goal is to promote transparency and address the ethical concerns surrounding authenticity.

BrushOrAlgorithm is a mobile application built with Flutter that classifies input images as AI or non-AI and provides a confidence score. It also generates a heatmap overlay to show which parts of the image influenced the model’s prediction.

Two models were trained for this task: one using a standard CNN algorithm and another using a GE-CNN algorithm. After evaluation, the GE-CNN model outperformed the CNN.

Colab Notebook of Model training and testing: https://colab.research.google.com/drive/1Jn6SkeCBGpRbyOPp5gpnd6bK3zswyNSn?usp=sharing
