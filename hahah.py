#neural network class definition
import numpy
import scipy.special
import csv
import matplotlib.pyplot
#%matplotlib inline
class neuralNetwork :
    # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input, hidden ,output lager
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #learning rate
        self.lr = learningrate
        #link weight matrices,wih and who
        #weight inside the arrary are w_i_j,where link is from node i to node j int the next layer
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #activation function is the sigmoid function
        self.activation_function = lambda  x:scipy.special.expit(x)
        pass
    #train the neural network
    def train(self,inputs_list,target_list):
        #convent inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin= 2).T
        targets =  numpy.array(target_list,ndmin = 2).T

        #calculate signals into hidden layer
        hidden_inputs =  numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)


        #output layer error is the (target-actual)
        output_errors = targets-final_outputs
        #hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #update the weight for the links between the hidden and output layers
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        # update the weight for the links between the input and hidden layers
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))

        pass
    #query the neural network
    def query(self,inputs_list):
        #CONVERT INPUT LIST TO 2D ARRAY
        inputs = numpy.array(inputs_list,ndmin= 2).T
        #CALCULATE SIGNALS INTO HIDDEN LAGER
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate= 0.1

n= neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

data_file= open("E:\Artificial_intelligence\MoreData\handwrite number\mnist_train.csv","r")
data_list= data_file.readlines()
data_file.close()
epochs = 5
for e in range(epochs):
    for record in data_list[1:]:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets =numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass
test_data_file =open("E:\Artificial_intelligence\DigitalRecognizerData\\test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard =[]
i=1
for record in test_data_list[1:]:
    all_values = record.split(',')
    '''correct_label = int(all_values[0])'''
    scaled_input = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    outputs = n.query(scaled_input)
    label = numpy.argmax(outputs)
    print(label,"network's answer")
    stu=[i,label]
    i=i+1
    out = open('forecast.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(stu)
    '''if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass'''
'''scorecard_array = numpy.asarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)'''
print("write over")