from LSTM import LSTM
import numpy as np

class RNN:
    # Constructor for RNN class
    def __init__ (self, x_inp, actual_op, hid_layer_dim, alpha, batch_size):

        # Data : input data
        self.x_inp = x_inp
        # Data : actual output
        self.actual_op = actual_op
        # Hyperparam : hidden layer size
        self.hid_layer_dim = hid_layer_dim
        # Hyperparam : learning rate
        self.alpha = alpha
        # Hyperparam : batch size
        self.batch_size = batch_size
        # input dimentions : number of input features
        self.inp_dim = x_inp.shape[1]
        # output dimension : number of output features
        self.op_dim = 1

        # 
        self.d_weight_mat = np.random.random(( self.op_dim, self.hid_layer_dim))
        # 
        self.updated_weight_mat = np.zeros_like(self.d_weight_mat)

        # 
        self.cell_state_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))
        # 
        self.predicted_outputs = np.zeros(( self.batch_size+1, self.op_dim))
        # 
        self.hid_state_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))

        # 
        self.forgetg_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))
        # 
        self.inputg_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))
        # 
        self.actg_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))
        # 
        self.outputg_vals = np.zeros(( self.batch_size+1, self.hid_layer_dim))

        # 
        self.d_weight_add = 1e-9
        self.grad_fact = 0.950
        self.diff_fact = 0.10
        # 
        self.LSTM = LSTM(self.inp_dim, self.hid_layer_dim, self.batch_size, self.alpha)

    # Helper Function 1 - Implemention of Sigmoid activation function 
    def sigmoid(self, inp):
        val = 1 / (1 + np.exp(-inp))
        return val

    # Helper Function 2 - Implemention of Derivative of Sigmoid function 
    def sigmoid_der(self, inp):
        der = self.sigmoid(inp) * (1 - self.sigmoid(inp))
        return der

    # Helper Function 3 - To update weights of weight matrix
    def update_weights(self, diff):
        self.updated_weight_mat = self.grad_fact * self.updated_weight_mat + self.diff_fact  * np.power(diff, 2)
        gradientVal = self.updated_weight_mat + self.d_weight_add
        self.d_weight_mat -= self.alpha/np.sqrt(gradientVal) * diff

    # 
    def fwd_pass(self, batch_x_inp):

        # 
        for t in range(1, batch_x_inp.shape[0]):
            # 
            self.LSTM.data_in = np.hstack((self.hid_state_vals[t-1], batch_x_inp[t]))

            cell_state_val, hidden_state_val, forgetg_val, inputg_val, actg_val, outputg_val = self.LSTM.fwd_pass()

            # 
            self.cell_state_vals[t] = cell_state_val
            self.hid_state_vals[t] = hidden_state_val
            self.forgetg_vals[t] = forgetg_val
            self.inputg_vals[t] = inputg_val
            self.actg_vals[t] = actg_val
            self.outputg_vals[t] = outputg_val
            # 
            self.predicted_outputs[t] = self.sig(np.dot(self.d_weight_mat, hidden_state_val))

        return self.predicted_outputs

    # 
    def bwd_pass(self, actual_batch_output, batch_x_inp):
        # 
        total_batch_error = 0
        # 
        cell_state_diff = np.zeros(self.hid_layer_dim)
        # 
        hidden_state_diff = np.zeros(self.hid_layer_dim)

        # 
        weight_difference = np.zeros(( self.op_dim, self.hid_layer_dim))
        # 
        total_forgetg_diff = np.zeros(( self.hid_layer_dim, self.inp_dim+self.hid_layer_dim))
        # 
        total_inputg_diff = np.zeros(( self.hid_layer_dim, self.inp_dim+self.hid_layer_dim))
        # 
        total_actg_diff = np.zeros(( self.hid_layer_dim, self.inp_dim+self.hid_layer_dim))

        total_outputg_diff = np.zeros(( self.hid_layer_dim, self.inp_dim+self.hid_layer_dim))

        # 
        for t in range(self.batch_size-1, -1, -1):
            # 
            curr_error = abs(self.predicted_outputs[t] - actual_batch_output[t])
            # 
            total_batch_error += curr_error
            # 
            weight_difference += np.dot(np.atleast_2d(curr_error * self.derivativeSig(self.predicted_outputs[t])),
                                        np.atleast_2d(self.hid_state_vals[t]))

            # 
            lstm_error = np.dot(curr_error, self.d_weight_mat)
            # 
            self.LSTM.data_in = np.hstack((self.hid_state_vals[t-1], batch_x_inp[t]))
            # 
            self.LSTM.cell_state_vals = self.cell_state_vals[t]
            # 
            forgetg_diff, inputg_diff, actg_diff, outputg_diff, cell_state_diff, hidden_state_diff = self.LSTM.bwd_pass(lstm_error, self.cell_state_vals[t-1], self.forgetg_vals[t], self.inputg_vals[t], self.actg_vals[t], self.outputg_vals[t], cell_state_diff, hidden_state_diff)

            # 
            total_forgetg_diff += forgetg_diff
            # 
            total_inputg_diff += inputg_diff
            # 
            total_actg_diff += actg_diff
            # 
            total_outputg_diff += outputg_diff

            # 
        self.LSTM.update_weight_mat(total_forgetg_diff/self.batch_size, total_inputg_diff/self.batch_size, total_actg_diff/self.batch_size, total_outputg_diff/self.batch_size)
            # 
        self.update_weights(weight_difference/self.batch_size)

        return total_batch_error

    

    # 
    def test(self, batch_x_inp, batch_y_op):

        # 
        batch_size = batch_x_inp.shape[0]
        # 
        self.acualOutput = batch_y_op
        # 
        self.cell_state_vals = np.zeros(( batch_size+1, self.hid_layer_dim))
        # 
        self.predicted_outputs = np.zeros(( batch_size+1, self.op_dim))
        # 
        self.hid_state_vals = np.zeros(( batch_size+1, self.hid_layer_dim))
        # 
        self.forgetg_vals = np.zeros(( batch_size+1, self.hid_layer_dim))
        # 
        self.inputg_vals = np.zeros(( batch_size+1, self.hid_layer_dim))
        # 
        self.actg_vals = np.zeros(( batch_size+1, self.hid_layer_dim))
        # 
        self.outputg_vals = np.zeros(( batch_size+1, self.hid_layer_dim))

        # 
        total_test_error = 0

        for t in range(1, batch_size):

            # 
            self.LSTM.data_in = np.hstack((self.hid_state_vals[t-1], batch_x_inp[t]))

            cell_state_val, hidden_state_val, forgetg_val, inputg_val, actg_val, outputg_val = self.LSTM.fwd_pass()

            # 
            self.cell_state_vals[t] = cell_state_val
            self.hid_state_vals[t] = hidden_state_val
            self.forgetg_vals[t] = forgetg_val
            self.inputg_vals[t] = inputg_val
            self.actg_vals[t] = actg_val
            self.outputg_vals[t] = outputg_val

            # 
            self.predicted_outputs[t] = self.sig(np.dot(self.d_weight_mat, hidden_state_val))

            curr_error = abs(self.predicted_outputs[t] - batch_y_op[t])
            total_test_error += curr_error

        return self.predicted_outputs, total_test_error
