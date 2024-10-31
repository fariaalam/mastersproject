from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class bilstm_classifier(nn.Module):

    """
     This class is for implementing Bi-LSTM  
    """ 
   
    def __init__(self, classes, ukp_embeddings):
        super().__init__()
        num_classes = classes
        embedding_dimension = 50
        hidden_dimension = 100
        number_of_layers = 1
        bidirectional = True
        batch_first = True
        number_of_embeddings = 400000

        self.embeddings = nn.Embedding(number_of_embeddings, embedding_dimension)
        self.embeddings.weight = torch.nn.Parameter(ukp_embeddings)
        self.embeddings.weight.requires_grad = False

        self.bilstm = nn.LSTM(input_size = embedding_dimension, hidden_size =hidden_dimension,
                                num_layers = number_of_layers, bidirectional = bidirectional, batch_first = batch_first)

        self.linear =nn.Linear(in_features =hidden_dimension*2, out_features = num_classes)

    def forward(self, inputs):
        """
           This function is for reducing Bi-lstm output dimension using linear model and generating class predictions using softmax
        """ 
        word_embs = self.embeddings(inputs)
        lstm_out, (hid_out , cell_out) = self.bilstm(word_embs)
        linear_output = self.linear(lstm_out)
        predictions = F.softmax(linear_output, dim = 1)
        return predictions


        
