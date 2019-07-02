# -*- coding: utf-8 -*- 
'''
Created on Tue Mar 26 10:37:38 2019 
 
@author: Kurros LAN

This is the main module of the Neural network(Vectorisation Version):
Main components:

'''
 
 
import numpy as np 
import time
import random
 


class NN_Vec: 
    ''' 
#Vectorized version of NN class 
Class variables :
1.neurons: taking a list of integers as input, each integer represents 
the number of neurons at that layer who's amount equals to the length of this list.
-----------------------------------------------------------
2.weight, bias: list of weight matrix and bias vectors.
-----------------------------------------------------------
3.z, act: z is the pre-activated value(affine combination) for
 the corresponding layer and act is the post-activated value.
-----------------------------------------------------------
4.loss_log,accuracy_log,lr_log: list of loss,accuracy and 
learning rate who are updated every 50 epochs.
'''

    def __init__(self, neurons): 
        self.neurons = neurons 
        self.n_layers = len(self.neurons) 
        self.weights = [] 
        self.bias = [] 
        self.act = [] 
        self.z = [] 
        self.i_log = [] 
        self.loss_log = [] 
        self.accuracy_log = [] 
        self.lr_log = []
        self.val_loss_log = []
        self.activate = self.act_sig 
        self.output_func = self.softmax 
        self.loss_func = self.loss_cross_entropy
        self.initialisation = self.initialisation_xavier
        self.forward = self.forward_normal
        self.measure = self.accuracy_test
 


    def initialisation_xavier(self,x,y):
        '''
        Xavier initialisation with random number genreted in the range between
        -1*4*(np.sqrt(6)/np.sqrt(dim_in+dim_out)) and 1*4*(np.sqrt(6)/np.sqrt(dim_in+dim_out))
        and bias are set to 0
       ''' 
        self.weights = [] 
        self.bias = [] 
        self.act = []
        A = -1*4*(np.sqrt(6)/np.sqrt(x.shape[1]+self.neurons[0]))
        B = 1*4*(np.sqrt(6)/np.sqrt(x.shape[1]+self.neurons[0]))
        self.weights.append(np.random.uniform(A, B, (x.shape[1], self.neurons[0]))) 
        self.bias.append(np.zeros((1, self.neurons[0])))
        for layer in range(1,len(self.neurons)):
            A = -1*4*(np.sqrt(6)/np.sqrt(self.weights[-1].shape[1]+ self.neurons[layer]))
            B = 1*4*(np.sqrt(6)/np.sqrt(self.weights[-1].shape[1]+ self.neurons[layer]))
            self.weights.append(np.random.uniform(A, B, (self.weights[-1].shape[1], self.neurons[layer]))) 
            self.bias.append(np.zeros((1, self.neurons[layer]))) 
        A = -1*4*(np.sqrt(6)/np.sqrt(self.neurons[-1] +  y.shape[1]))
        B =  1*4*(np.sqrt(6)/np.sqrt(self.neurons[-1] +  y.shape[1]))
        self.weights.append(np.random.uniform(A, B, (self.neurons[-1], y.shape[1]))) 
        self.bias.append(np.zeros((1, y.shape[1]))) 

       
    def forward_normal(self, weights, bias, x): 
        '''
        forward and backwards propagation, 
        taking an N*D matrix of examples (N = amount of examples, D = dimension of examples) as input, 
        they are vectorized version of the same function in the NN_Non_Vec.py.
        ''' 
        curr_input = x 
        curr_output = x 
        self.act = [x] 
        self.z = [] 
        counter = 0 
        for w, b in zip(weights, bias): 
            curr_input = np.dot(curr_output,w) + b 
            self.z.append(curr_input) 
            counter += 1 
            if counter < len(weights): 
                curr_output = self.activate(curr_input) 
            else: 
                curr_output = self.output_func(curr_input) 
            self.act.append(curr_output) 
        return curr_output 


   

    def act_sig(self, x, mode='ford'): 
        '''
        sigmoid function,back = derivative
        '''
        if mode == 'ford': 
            result = 1. / (1. + np.exp(-x)) 
        elif mode == 'back': 
            result = self.act_sig(x) * (1 - self.act_sig(x)) 
            #result = 1. / (1. + np.exp(-1 * x)) * (1. - (1. / (1. + np.exp(-1 * x)))) 
        return result 
    

    def softmax(self, x, mode='ford'): 
        '''
        vectorized softmax:derivative of softmax for an 1*D vectors is an D*D jacobian matrix
        this function takes a matrix of N vectors and return a ND*D jacobian matrix
        '''
        exp = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        if mode == 'ford': 
            return exp / exp.sum(axis=1, keepdims=True) 
        if mode == 'back': 
            ford = self.softmax(x, mode='ford') 
            step = ford.shape[1] 
            ford = ford.reshape(-1,1) 
            dim = ford.shape[0] 
            block = np.dot(ford[0:step],ford[0:step].T) 
            diag = np.diagflat(ford[0:step]) 
            block = diag - block 
            for i in range(step,dim,step): 
                sub = ford[i:i+step] 
                diag = np.diagflat(sub) 
                next_block = diag - np.dot(sub,sub.T) 
                block = np.concatenate((block,next_block)) 
            return block 
        


    def loss_cross_entropy(self, y_gold, y_pred, mode='ford'): 
        '''
        ATTENTION:Only accept a one-hot encoded y_gold as input
        '''
 
        if mode == 'ford': 
            return np.sum((-1*y_gold)*np.log(y_pred)) 
        if mode == 'back': 
            with np.errstate(divide = 'ignore'): 
                nozeros = (-1 / (y_pred*y_gold)) 
                nozeros[nozeros == -np.inf] = 0 
            return nozeros 
    
 
    def multi_dot(self,dl_da,da_dz): 
        '''
        this function mainly for the first step of backpropagation,where 
        the derivative of softmax is not a square matrix(normaly it is,but here it takes a matrix as input),
        so it's not compatible with the dot product
        this function takes an ND*D matrix (dlda) and an N*D matrix and output 
        and N*D matrix (dadz), that means every D line of dlda(an N*D matrix) 
        will be multiplied by 1 line of dadz (a 1*D vector)
        it probably will misbehave when the last layer is not softmax...
        '''   
        axe_x = dl_da.shape[0] 
        axe_y = dl_da.shape[1] 
        dlda = dl_da.reshape(axe_x,1,axe_y) 
        dadz = da_dz.reshape(axe_x,axe_y,axe_y) 
        dot = np.einsum('ijp,ipj->ij',dlda,dadz) 
        return dot 
             
      
    def backward(self, initial_grad, weights, bias): 
        '''
        three variable are calculated at each layer:
            da_dz,dl_dz and dl_dw
        '''  
        dl_da = initial_grad 
        weights_grad = [] 
        bias_grad = [] 
        inverse_weights = weights[::-1] 
        inverse_act = self.act[::-1] 
        inverse_z = self.z[::-1] 
        for layer in range(len(weights)): 
            if layer == 0 : 
                da_dz = self.output_func(inverse_z[layer],'back') 
                dl_dz = self.multi_dot(dl_da,da_dz) 
            else: 
                da_dz = self.activate(inverse_z[layer], 'back') 
                dl_dz = (np.multiply(dl_da.T, da_dz)) 
            dl_da = np.dot(inverse_weights[layer],dl_dz.T) 
            weights_grad.append((np.dot(dl_dz.T, inverse_act[layer + 1])).T) 
            bias_grad.append(np.sum(dl_dz,axis =0,keepdims = True)) 
        return weights_grad[::-1], bias_grad[::-1] 
 
 
    def predit(self, x): 
        return self.forward(self.weights, self.bias,x) 
  

    def mini_batch_train(self,X,Y,batch_size,epoch,learning_rate = 0.1,decay = 1,beta = 0 ,validation = False,x_val = 0 ,y_val = 0): 
        '''
        the core function, other training functions in this class is for experiment purpose
        the decay rate for decreasing learning rate at every x epoch
        beta: parameter for the momentum, momentum will be updated at every batch 
        and set to zero at the beginning of each epoch
        at every x epoch, the performance of the network (loss on the training set, 
        accuracy and loss on the validation set ) will be recorded 
        ''' 
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [0] 
        self.i_log = [] 
        self.lr_log = [] 
        self.initialisation(X, Y) 
        for i in range(epoch): 
            if i%25 == 0 and i !=0: 
                learning_rate = (1./(1. + decay*i))*learning_rate 
                #learning_rate = learning_rate*0.1
                print('Total loss at epoch', i,loss) 
                print('Diffrence:',loss - self.loss_log[-1]) 
                self.loss_log.append(loss) 
                loss = 0 
                self.i_log.append(i) 
                if validation != False:
                    accuracy,val_loss = self.accuracy_test(x_val,y_val) 
                    self.val_loss_log.append(val_loss)
                    self.accuracy_log.append(accuracy)
                    print('Accuracy on val:' ,accuracy)
                    print('Loss on val:',val_loss)
                    print('---------------------------------')
                if decay != 0: 
                    self.lr_log.append(learning_rate) 
            # initialise list for momentum
            v_w = [0 for i in range(len(self.weights))] 
            v_b = [0 for i in range(len(self.weights))] 
            #Batch training begins here
            for i in range(0,X.shape[0],batch_size): 
                #slicing the big training set
                slice_x = X[i:i+batch_size] 
                slice_y = Y[i:i+batch_size] 
                weights, bias = self.weights,self.bias 
                y_pred = self.forward(weights, bias,slice_x) 
                batch_loss = 1/X.shape[0]*self.loss_func(slice_y, y_pred,'ford') 
                initial_grad = self.loss_func(slice_y, y_pred,'back') 
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias)
                #update the moumentum weights and bias 
                for j in range(len(self.weights)):
                    v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*v_w[j] 
                for k in range(len(self.bias)): 
                    v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*v_b[k] 
                #cumulate loss for each mini-batch
                loss += batch_loss 
                
   
    def stocha_train_early_stopping(self,X,Y,batch_size,stop_condition_1 = 0.9,stop_condition_2 = 50000,learning_rate = 0.1,random_LR = False,decay = 0,beta = 0 ,x_val = 0 ,y_val = 0): 
        '''
        train on random mini-batch(10-50 examples) sampling from the trainning set
        it only updates weights and bias when the accuracy on the validation set is improved
        it doesn't take a fixed epoch as a parameter, instead, the loop will continue infinitely until one of two stop condition is met
        stop_condition_1 is the minimal  accuracy rate, the function will stop when its accuracy is bigger than this parameter
        stop_condition_2 is the maximum amount of iterations who 'fail' to update weights and bias, the function will stop when the variable 'stop_counter' is bigger than this parameter
        stop_counter will +1 when the function fails to update and -1 when it successfully updates weights and bias
        if random_LR is set to True, when the update fails,the function will randomly pick a learning rate between 0.01 and 0.2
        '''  
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [0] 
        self.i_log = [] 
        self.lr_log = [] 
        self.initialisation(X, Y)
        new_val_loss = 0
        new_accuracy = 0
        i = 0
        v_w = [0 for i in range(len(self.weights))]
        v_b = [0 for i in range(len(self.weights))]
        stop_counter = 0
        while True:
            if i == 0:
                accuracy,val_loss = self.accuracy_test(x_val,y_val) 
            i +=1
            if decay != 0: 
                self.lr_log.append(learning_rate)
            if i%25 == 0 and i !=0: 
                #learning_rate = (1./(1. + decay*i))*learning_rate 
                #learning_rate = learning_rate*0.1
                print('Total loss at epoch', i,loss) 
                print('Diffrence:',loss - self.loss_log[-1]) 
                self.loss_log.append(loss) 
                loss = 0 
                self.i_log.append(i) 
                self.val_loss_log.append(val_loss)
                self.accuracy_log.append(accuracy)
                print('Accuracy on val:' ,accuracy)
                print('Loss on val:',val_loss)
                print('---------------------------------')
            sample_idxs = np.unique(np.random.randint(X.shape[0], size= batch_size))
            x_samples = X[sample_idxs]
            y_samples = Y[sample_idxs]
            weights, bias = self.weights,self.bias
            weights_buffer,bias_buffer = self.weights,self.bias
            momentum_bias = v_b
            momentun_weights = v_w
            y_pred = self.forward(weights, bias,x_samples) 
            batch_loss = 1/X.shape[0]*self.loss_func(y_samples, y_pred,'ford') 
            initial_grad = self.loss_func(y_samples, y_pred,'back') 
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias)
            for j in range(len(self.weights)):
                v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                self.weights[j] = self.weights[j] - learning_rate * 1/x_samples.shape[0]*v_w[j] 
            for k in range(len(self.bias)): 
                v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                self.bias[k] = self.bias[k] - learning_rate *1/x_samples.shape[0]*v_b[k]
            new_accuracy,new_val_loss = self.accuracy_test(x_val,y_val)
            if new_accuracy < accuracy:
                stop_counter += 1
                v_b = momentum_bias
                v_w = momentun_weights
                self.weights = weights_buffer
                self.bias = bias_buffer
                if random_LR == True:
                    learning_rate = random.uniform(0.01,0.2)
            else:
                stop_counter -= 1
                accuracy = new_accuracy
                val_loss = new_val_loss
            loss += batch_loss #cumulate loss for each mini-batch
            if accuracy >= stop_condition_1:
                print('Training Complete,final accuracy rate :' + accuracy)
                break
            if stop_counter >stop_condition_2:
                print('Failed to converge,final accuracy rate :' + accuracy)
                break
         
 
    def accuracy_test(self,X,Y,onehot = True,boundary = 0.5): 
        result = self.predit(X)
        sample_loss = 1/X.shape[0] * self.loss_func(Y,result)
        if onehot == True: 
            result[result >= boundary] = 1 
            good = np.sum(Y.reshape(result.shape[0],result.shape[1]) == result) 
            score = float(good)/float(Y.shape[0]) 
        return score,sample_loss



    def score (self,X,Y,boundary = 0.5):

        '''
        good_known : amount of correctly tagged known words (not UNK)
        y_known.shape[0] : amount of all known words
        good_UNK :amount of correctly tagged UNK words
        y_UNK.shape[0] :mount of all UNK words
        19124 : since in our embedding, the vector for word 'UNK' starts with 1.9124
        it's a trick for saving time, but it also makes this function totally not general
        '''
        x_UNK = X[np.where((X[:,0]*10000).astype('int') == 19124)]
        y_UNK = Y[np.where((X[:,0]*10000).astype('int') == 19124)]
        x_known = X[np.where((X[:,0]*10000).astype('int') != 19124)]
        y_known = Y[np.where((X[:,0]*10000).astype('int') != 19124)]
        known_pred = self.predit(x_known)
        UNK_pred = self.predit(x_UNK)
        known_pred[known_pred >= boundary] = 1
        UNK_pred[UNK_pred >=boundary] = 1
        good_known = np.sum(y_known.reshape(known_pred.shape[0],known_pred.shape[1]) == known_pred)
        good_UNK = np.sum(y_UNK.reshape(UNK_pred.shape[0],UNK_pred.shape[1]) == UNK_pred)
        return good_known,y_known.shape[0],good_UNK,y_UNK.shape[0]
        
        

    def batch_train(self,X,Y,epoch,learning_rate,beta = 0,x_val = 0,y_val = 0):
        '''
        train on the entirety of the training set, it can be used with GPU in order to 
        minimise the data transferer time
        however, I've tested it using Cupy, the result is underwhelming
        '''
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [] 
        self.i_log = [] 
        self.initialisation(X, Y) 
        for i in range(epoch): 
            v_w = [0 for i in range(len(self.weights))] 
            v_b = [0 for i in range(len(self.weights))] 
            if i%50 == 0 and i !=0: 
                if x_val !=0 and y_val !=0: 
                    self.accuracy_log(self.accuracy_test(x_val,y_val)) 
                    print(self.accuracy_log[-1]) 
                self.loss_log.append(loss) 
                #loss
                self.i_log.append(i) 
                print('Total loss at epoch', i,loss) 
            weights, bias = self.weights,self.bias 
            y_pred = self.forward(weights, bias,X) 
            loss =  self.loss_func(Y, y_pred,'ford').sum() 
            initial_grad = self.loss_func(Y, y_pred,'back') 
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
            for j in range(len(self.weights)): 
                v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j] 
                self.weights[j] = self.weights[j] - learning_rate * 1/X.shape[0]*weights_grad[j] 
            for k in range(len(self.bias)): 
                v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                self.bias[k] = self.bias[k] - learning_rate *1/X.shape[0]*bias_grad[k] 
            loss += 1/X.shape[0] * loss 
  
            
            
            
    
 

    def mini_batch_train_no_sparse(self,training_set,batch_size,epoch,learning_rate,beta = 0, validation = 0):
        '''
        the same function but accept a single dense matrix (N*D+1), 
        at every mini-batch this matrix, will be converted into 2 matrix :
        X and Y where Y is the sparse matrix for the tag
        ''' 
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [] 
        self.i_log = [] 
        x_train = training_set[:,:-1] 
        y_train = training_set[:,-1] 
        max_dim = int(np.max(y_train))+1 
        self.initialisation(x_train, np.zeros((1,max_dim))) 
        v_w = [0 for i in range(len(self.weights))] 
        v_b = [0 for i in range(len(self.weights))] 
        for i in range(epoch): 
            if i%50 == 0 and i !=0: 
                if validation != 0: 
                    x_val = validation[:,:-1] 
                    y_val = validation[:,-1] 
                    self.accuracy_log.append(self.accuracy_test(x_val,y_val)) 
                    print(self.accuracy_log[-1]) 
                self.loss_log.append(loss) 
                self.i_log.append(i) 
                print('Total loss at epoch', i,loss) 
            for i in range(0,x_train.shape[0],batch_size): 
                slice_x = x_train[i:i+batch_size] 
                index = y_train[i:i+batch_size].astype('int') 
                slice_y = np.zeros((index.shape[0],max_dim)) 
                slice_y[np.arange(index.shape[0]),index] = 1. 
                weights, bias = self.weights,self.bias 
                y_pred = self.forward(weights, bias,slice_x) 
                batch_loss = 1/slice_x.shape[0] * self.loss_func(slice_y, y_pred,'ford') 
                initial_grad = self.loss_func(slice_y, y_pred,'back') 
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
                for j in range(len(self.weights)): 
                    v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*v_w[j] 
                for k in range(len(self.bias)): 
                    v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*v_b[k] 
                loss +=  batch_loss 
 
    
    def input_parameters(self,weights,bias): 
        self.weights = weights 
        self.bias = bias 

     
    def learning_rate_finder(self,X,Y,beta = 0,initial_rate = 1e-6,multiplier = 1.1,stop = 100,sample_size = 50): 
        '''
        generally increasing the learning rate from initial_rate and return the corresponding accuracy
        and loss on the validation set
        using for determinate the optimal learning rate as well as the range of learning rate 
        for the Clr_train function
        '''    
        print('--Training begins--') 
        self.initialisation(X,Y) 
        loss_log = [] 
        learning_rate_log = []
        accuracy_log = []
        learning_rate = initial_rate 
        runs = 0 
        while True:  
            v_w = [0 for i in range(len(self.weights))] 
            v_b = [0 for i in range(len(self.weights))] 
            sample_idxs = np.unique(np.random.randint(X.shape[0], size= sample_size)) 
            x_samples = X[sample_idxs] 
            y_samples = Y[sample_idxs] 
            weights, bias = self.weights,self.bias 
            y_pred = self.forward(weights, bias,x_samples)
            accuracy_log.append(self.accuracy_test(x_samples,y_samples)[0])
            batch_loss = 1/x_samples.shape[0]*self.loss_func(y_samples, y_pred,'ford') 
            initial_grad = self.loss_func(y_samples, y_pred,'back') 
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
            for j in range(len(self.weights)): 
                v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                self.weights[j] = self.weights[j] - learning_rate * 1/x_samples.shape[0]*v_w[j] 
            for k in range(len(self.bias)): 
                v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                self.bias[k] = self.bias[k] - learning_rate *1/x_samples.shape[0]*v_b[k] 
            learning_rate = learning_rate * multiplier 
            learning_rate_log.append(learning_rate) 
            loss_log.append(batch_loss) 
            runs +=1 
            stop_condition = batch_loss > 2 * loss_log[0] 
            if (stop_condition and len(loss_log) > 20) or runs>stop: 
                return learning_rate_log,loss_log,accuracy_log
            
        
    def Clr_train(self,X,Y,batch_size,epoch,learning_rate_0 = 0.01,upper_bound = 0.2,beta = 0 ,x_val = 0 ,y_val = 0): 
        '''
        Experiment with CLR (Cyclical Learning Rates) method
        the learning rate will cycle through learning_rate_0 and upper_bound during one epoch,
        ideally, the learning rate will cycle back to learning_rate_0 at the end of each epoch
        '''   
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [0] 
        self.i_log = [] 
        self.lr_log = [] 
        self.initialisation(X, Y) 
        iteration =float( X.shape[0]/batch_size)
        step = 2*iteration
        
        for i in range(epoch): 
            mini_batch_count = 0
            if i%25 == 0 and i !=0: 
                print('Total loss at epoch', i,loss) 
                print('Diffrence:',loss - self.loss_log[-1]) 
                self.loss_log.append(loss)
                loss = 0 
                self.i_log.append(i) 
                if True:
                    accuracy,val_loss = self.accuracy_test(x_val,y_val) 
                    self.val_loss_log.append(val_loss)
                    self.accuracy_log.append(accuracy)
                    print('Accuracy on val:' ,accuracy)
                    print('Loss on val:',val_loss)
                    print('---------------------------------')
            v_w = [0 for i in range(len(self.weights))] 
            v_b = [0 for i in range(len(self.weights))] 
            for i in range(0,X.shape[0],batch_size): 
                mini_batch_count += 1
                #cycle =  np.float((1+ep_count/2*step))
                #x = np.abs((ep_count/step) - (2*cycle) + 1)
                #learning_rate = learning_rate_0 + (upbound - learning_rate_0) * max(0,(1-x))
                cycle = np.floor(1+mini_batch_count/(2*step))
                x = np.abs(mini_batch_count/step - 2*cycle + 1)
                learning_rate = learning_rate_0 + (upper_bound-learning_rate_0)*np.maximum(0, (1-x))/float(2**(cycle-1)) 
                slice_x = X[i:i+batch_size] 
                slice_y = Y[i:i+batch_size] 
                weights, bias = self.weights,self.bias
                y_pred = self.forward(weights, bias,slice_x) 
                batch_loss = 1/X.shape[0]*self.loss_func(slice_y, y_pred,'ford') 
                initial_grad = self.loss_func(slice_y, y_pred,'back') 
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
                for j in range(len(self.weights)): 
                    v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*v_w[j] 
                for k in range(len(self.bias)): 
                    v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*v_b[k] 
                loss += batch_loss 
  
 
'''   
toy set (XOR) for verifying the NN is working correctly

x = np.array([[1., 0.],[0.,1.],[1.,1.],[0.,0.],[1., 0.]]) 
y = np.array([[1., 0.],[1.,0.],[0.,1.],[0.,1.],[1., 0.]]) 
 
st = time.time() 
nn = NN_Vec([50])  
nn.batch_train(x, y,6000,0.1) 
prd = nn.predit(x)
print(prd,y) 
ed = time.time() 
print(ed-st) 
'''