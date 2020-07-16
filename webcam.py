import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# define the network class again... f you torch
class Network(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1 = nn.Conv2d(1, 10, 3)
    self.pool1 = nn.MaxPool2d(2)
    
    self.conv2 = nn.Conv2d(10, 20, 3)
    self.pool2 = nn.MaxPool2d(2)
    
    self.conv3 = nn.Conv2d(20, 30, 3) 
    self.dropout1 = nn.Dropout2d()
    
    self.fc3 = nn.Linear(30 * 3 * 3, 270) 
    self.fc4 = nn.Linear(270, 26) 
    
    self.softmax = nn.LogSoftmax(dim=1)
      
  def forward(self, x):
    # Pass the input tensor through each of our operations
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool2(x)
    
    x = self.conv3(x)
    x = F.relu(x)
    x = self.dropout1(x)
            
    x = x.view(-1, 30 * 3 * 3) 
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    
    return self.softmax(x)
      
  def test(self, predictions, labels):
      
    self.eval()
    correct = 0
    for p, l in zip(predictions, labels):
      if p == l:
        correct += 1
      
    acc = correct / len(predictions)
    print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))
      

  def evaluate(self, predictions, labels):
    correct = 0
    for p, l in zip(predictions, labels):
      if p == l:
        correct += 1
    
    acc = correct / len(predictions)
    return(acc)

# ==========================================


# mapping the alphabet
alphabets = [
  'a', 'b', 'c', 'd', 'e',
  'f', 'g', 'h', 'i', 'j',
  'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't',
  'u', 'v', 'w', 'x', 'y',
  'z'
]

# load model
model = torch.load('./models/SGD_model.pt')

# get camera
camera = cv2.VideoCapture(0)

# keep looping 
while True:
  # Grab the current from webcam
  (grabbed, frame) = camera.read()
  # flip the image
  frame = cv2.flip(frame, 1)
  # change to greyscale image and resize to 28 * 28
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (28, 28))
  image = image.reshape(1, 28, 28)

  model_out = model(Variable(torch.FloatTensor([image.tolist()])))
  predict = torch.max(model_out.data, 1)[1].numpy()[0]
  predict_alphabet= alphabets[predict]
  print(predict_alphabet)

  # show the webcam
  cv2.imshow("Tracking", frame)

  # wait for ESC key
  key = cv2.waitKey(20)

  # Check to see if we have reached the end of the video (useful when input is a video file not a live video stream)
  if not grabbed or key == 27:
    break   
