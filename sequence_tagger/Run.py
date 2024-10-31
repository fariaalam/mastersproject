import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

from raw_data_processing import (
    read_data_file, 
    read_embeddings_file,
    word2index,
    label2index,
    data4model,
    plot_outOfVocab,
    findUniqueLabels
)
from model import bilstm_classifier
number_of_epochs = 20


def training(trainData, trainLables):
    for epoch in range(number_of_epochs):
        iteration = 0
        bilstm.train()
        for data in zip(trainData, trainLables):
            iteration = iteration + 1
            batch_train_data, batch_train_labels = data
            batch_train_data = torch.LongTensor([batch_train_data])
            batch_train_labels = torch.LongTensor(batch_train_labels)
            output = bilstm(batch_train_data)
            loss =  F.cross_entropy(output[0], batch_train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 500 == 0:
                output_string = "Training epoch : "+ str(epoch) + " iteration : " + str(iteration) + " loss : " + str(loss.item())
                print(output_string)

            #if iteration == 50 :
            #    break

        validation(dev_data,dev_labels, best_validation_accuracy)


def validation(devData, devLabels, best_validation_accuracy):
    iteration = 0
    bilstm.eval()
    predicted_labels = []
    true_labels = []
    for data in zip(devData, devLabels):
        iteration = iteration + 1
        batch_dev_data, batch_dev_labels = data
        true_labels.extend(batch_dev_labels)
        batch_dev_data = torch.LongTensor([batch_dev_data])
        batch_dev_labels = torch.LongTensor(batch_dev_labels)
        output = bilstm(batch_dev_data)
        loss =  F.cross_entropy(output[0], batch_dev_labels)
        predictions = torch.max(output[0],1)[1]
        predicted_labels.extend(predictions.tolist())
       
        if iteration %  500 == 0:
            output_string = " Validation iteration : " + str(iteration) + " loss : " + str(loss.item())
            print(output_string)
        #if iteration == 50 :
        #    break
    #calculating macro and micro averaged F1 score for dev data
    f1_macro_average = f1_score(true_labels, predicted_labels, average = 'macro')
    accuracy = accuracy_score(true_labels,predicted_labels)
    f1_micro_average = f1_score(true_labels, predicted_labels, average = 'micro')
     
    performance_score = "F1 macro-avg : " + str(f1_macro_average)+ " " + "F1 micro-avg : " + str(f1_micro_average)+ " " + "Accuracy : " + str(accuracy)
    print("For Dev Data : " + performance_score)
    
    #Saving the Best model
    if accuracy > best_validation_accuracy:
        global best_model 
        best_model = bilstm
        best_validation_accuracy = accuracy
        
   

def test(testData, testLabels):
    iteration = 0
    best_model.eval()
    predicted_labels = []
    true_labels = []
    for data in zip(testData, testLabels):
        iteration = iteration + 1
        batch_test_data, batch_test_labels = data
        true_labels.extend(batch_test_labels)
        batch_test_data = torch.LongTensor([batch_test_data])
        batch_test_labels = torch.LongTensor(batch_test_labels)
        output = best_model(batch_test_data) # using the best model found using dev data
       
        predictions = torch.max(output[0],1)[1]
        predicted_labels.extend(predictions.tolist())
       
        #if iteration == 50 :
        #    break
    #Calculating macro and micro averaged F1 score for test data

    f1_macro_average = f1_score(true_labels, predicted_labels, average = 'macro')
    accuracy = accuracy_score(true_labels,predicted_labels)
    f1_micro_average = f1_score(true_labels, predicted_labels, average = 'micro')
    
    performance_score = "F1 macro-avg : " + str(f1_macro_average)+ " " + "F1 micro-avg : " + str(f1_micro_average)+ " " + "Accuracy ; " + str(accuracy)
    print("For Test Data : " +performance_score)
    
    #Calculating Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    confusion_display = ConfusionMatrixDisplay(confusion_matrix =conf_matrix, display_labels= unique_labels) 
    confusion_display.plot(values_format = '')

    plt.title("Confusion Matrix of Test data")
    plt.xticks(rotation = 30)
    plt.savefig("Confusion Matrix of Test data", bbox_inches = 'tight')
    plt.close()
    



words, embeddings = read_embeddings_file("glove.6B.50d.txt")
data, labels = read_data_file("data/train.conll")  
unique_labels = findUniqueLabels(labels)
numeric_labels = list(range(len(unique_labels)))
w2i = word2index(words)
l2i = label2index(unique_labels)
train_data, train_labels, out_of_vocab_lables, total_labels = data4model(data, labels, w2i, l2i, unique_labels)


data, labels = read_data_file("data/dev.conll")  
dev_data, dev_labels, _, _ = data4model(data, labels, w2i, l2i, unique_labels)

data, labels = read_data_file("data/test.conll")  
test_data, test_labels, _, _ = data4model(data, labels, w2i, l2i, unique_labels)

embeddings = torch.tensor(embeddings)
bilstm = bilstm_classifier(len(unique_labels), embeddings)

optimizer = torch.optim.Adam(list(bilstm.parameters()), lr = 0.001)
best_validation_accuracy = 0
best_model = None
training(train_data, train_labels)
test(test_data,test_labels)
plot_outOfVocab(out_of_vocab_lables,total_labels,unique_labels)