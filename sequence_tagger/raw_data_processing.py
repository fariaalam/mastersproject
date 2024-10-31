
import numpy as np 
import matplotlib.pyplot as plt 




def read_data_file(data_file_path):
    """
    This function is for reading the data file
     
    args: 
        data_file_path : a string which contains the path of the file

    returns:
        data : contains the sentences as a list of words
        label : contains the lables of each word  of each sentence as a list
        unique_labels: contains the unique label present in the dataset 
        
    """
    file_data = open(data_file_path, "r")
    data = []
    labels = []
    entry_words = []
    entry_label = []
    flag = 0
    for entry in file_data :
        entry = entry.strip()
        entry = entry.split()
        if len(entry) != 0 : 
            # I dont want to consider the first line containing the column names
            if entry[0] == "-DOCSTART-" :
                continue
        if len(entry) > 0 :
            entry_words.append(entry[0].lower())
            entry_label.append(entry[3])
        
        elif len(entry) == 0 and flag == 0  :
            flag = 1    
            continue
        elif len(entry) == 0 and flag == 1 : 
            data.append(entry_words)
            labels.append(entry_label) 
            entry_words = []
            entry_label = []  

    return data, labels
            


def read_embeddings_file(embedding_file_path):
    """
    This Function is for reading word embeddings from file and convert it into list

    arg:
        The path of the embedding file

    return:
        It returns lists of words and embeddings        

    """
    embedding_file = open(embedding_file_path, 'r')
    words = []
    embeddings = []
    for entry in embedding_file:
        entry = entry.strip()
        entry = entry.split()
        words.append(entry[0]) 
        temp_embedding = entry[1:]
        embedding = [float(i) for i in temp_embedding]
        embeddings.append(embedding)

    return words, embeddings


def word2index(words):
    """
     This function is for indexing each word by creating dictionary
    """ 
    w2i = dict()
    for index , word  in enumerate(words):
        w2i[word]= index
    return w2i

def label2index(labels):
    """
     This function is for indexing each label by creating dictionary
    """ 
    l2i = dict()
    for index , label  in enumerate(labels):
        l2i[label]= index
    return l2i

def data4model(data, labels, w2i, l2i, unique_labels):
    """
     This function is for replacing data and labels with the indices

     arg:
          word and corresponding labels with their individual dictionary

     return:
          list of word indices, labels indices, out of vocabulary words and total number of words per label      
    
    """ 
    data_indices = []
    labels_indices = []
    oov_labels = [0] * len(unique_labels)
    total_labels = [0] * len(unique_labels)
    oov = 0

    for i in range(len(data)):
        row = data[i]
        label = labels[i]
        row_indices = []
        label_indices = []
        for j in range(len(row)):
            w = row[j]
            l = label[j]
            total_labels[l2i[l]] = total_labels[l2i[l]] + 1
            try:
                row_indices.append(w2i[w])
                label_indices.append(l2i[l])
            except Exception as ex:
                oov_labels[l2i[l]] = oov_labels[l2i[l]] + 1
                oov = oov + 1
                continue
        if len(row_indices) > 0:
            data_indices.append(row_indices) 
            labels_indices.append(label_indices)    


    return data_indices, labels_indices, oov_labels, total_labels

def plot_outOfVocab(oov_labels, total_labels, unique_labels):
    """
     This function is for plotting total number of words per label out of vocabulary words per label

    """ 
    X_axis = np.arange(len(unique_labels))
    plt.bar(X_axis - 0.2, total_labels , 0.4, label = 'Total Number of Occurances per Label')
    #plt.bar(X_axis + 0.2,oov_labels , 0.4, label = 'Out of Vocabulary Occurances per Label')
    plt.xticks(X_axis,unique_labels )
    plt.xlabel("Unique Labels ")
    plt.ylabel("Number of Occurances")
    plt.title("Total Occurances per Label")
    plt.legend()
    plt.savefig("Total Occurances per Label", bbox_inches = 'tight')
    plt.close()
    
    

    X_axis = np.arange(len(unique_labels))
    #plt.bar(X_axis - 0.2, total_labels , 0.1, label = 'Total Number of Occurances per Label')
    plt.bar(X_axis + 0.2,oov_labels , 0.4, label = 'Out of Vocabulary Occurances per Label')
    plt.xticks(X_axis,unique_labels )
    plt.xlabel("Unique Labels ")
    plt.ylabel("Number of Occurances")
    plt.title("Out of Vocabulary Occurances per Label")
    plt.legend()
    plt.savefig("Out of Vocabulary Occurances per Label", bbox_inches = 'tight')
    plt.close()



def findUniqueLabels(labels):
    """
     This function is for finding total number of unique labels
    """ 
    unique_labels = []
    for label in labels:
        for l in label:
            if l not in unique_labels:
                unique_labels.append(l)       
    
    return unique_labels

