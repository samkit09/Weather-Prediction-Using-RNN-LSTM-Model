import numpy as np

class LSTM:
    # Constructor for LSTM class
    def __init__ (self, inp_dim, out_dim, batch_size, alpha):
        self.data_in = np.zeros(inp_dim + out_dim)   # Data : input data
        self.data_out = np.zeros(shape = (1, out_dim))  # Data : output data
        self.batch_size = batch_size  # Hyperparam : batch size
        self.cell_state_vals = np.zeros(out_dim)
        self.alpha = alpha  # Hyperparam : learning rate
        self.input_weight_mat = np.random.random((out_dim, inp_dim + out_dim))   # input weights
        self.forget_weight_mat = np.random.random((out_dim, inp_dim + out_dim))  # forget weights
        self.output_weight_mat = np.random.random((out_dim, inp_dim + out_dim))  # output weights
        self.activation_weight_mat = np.random.random((out_dim, inp_dim + out_dim))  # activation weights
        self.update_forget_weight_mat = np.zeros_like(self.forget_weight_mat)
        self.update_input_weight_mat = np.zeros_like(self.input_weight_mat)
        self.update_activation_weight_mat = np.zeros_like(self.activation_weight_mat)
        self.update_output_weight_mat = np.zeros_like(self.output_weight_mat)

    # Helper Function 1 - Implementation of Sigmoid activation function
    def sigmoid(self, inp):
        val = np.exp(inp) / (1 + np.exp(inp))
        return val
    
    # Helper Function 2 - Implementation of Derivative of Sigmoid function
    def sigmoid_der(self, inp):
        der = self.sigmoid(inp) * (1 - self.sigmoid(inp))
        return der
    
    # Helper Function 3 - Implementation of Derivative of hyperbolic tangent
    def tanh_der(self, inp):
        return 1 - np.tanh(inp) * np.tanh(inp)

    # forward pass
    def fwd_pass(self):
        input_value = self.sigmoid(np.dot(self.input_weight_mat, self.data_in))
        forget_value = self.sigmoid(np.dot(self.forget_weight_mat, self.data_in))
        output_value = self.sigmoid(np.dot(self.output_weight_mat, self.data_in))
        activation_value = np.tanh(np.dot(self.activation_weight_mat, self.data_in))
        self.cell_state_vals = self.cell_state_vals*forget_value + (activation_value*input_value)
        self.data_out = output_value * np.tanh(self.cell_state_vals)

        return self.cell_state_vals, self.data_out, forget_value, input_value, activation_value, output_value

    # backward pass
    def bwd_pass(self, err_RNN, prev_cell_state, forget_value, input_value, activation_value, output_value, del_cell_state, del_hidden_state):
        err_RNN = err_RNN + del_hidden_state
        der_output = np.tanh(self.cell_state_vals) * err_RNN
        der_cell_state = err_RNN * output_value * self.tanh_der(self.cell_state_vals) + del_cell_state
        del_prev_cell_state = der_cell_state * forget_value
        der_forget_val = der_cell_state * prev_cell_state
        der_input_val = der_cell_state * activation_value
        der_activation_val = der_cell_state * input_value

        in_2d = np.atleast_2d(self.data_in)
        out_val = np.atleast_2d(der_output * self.sigmoid_der(output_value)).T
        del_output_val = np.dot(out_val, in_2d)
        forget_val = np.atleast_2d(der_forget_val * self.sigmoid_der(forget_value)).T
        del_forget_val = np.dot(forget_val, in_2d)
        in_val = np.atleast_2d(der_input_val * self.sigmoid_der(input_value)).T
        del_input_val = np.dot(in_val , in_2d)
        act_val = np.atleast_2d(der_activation_val * self.tanh_der(activation_value)).T
        del_activation_val = np.dot(act_val, in_2d)
        del_prev_hidden_state = np.dot(der_activation_val, self.activation_weight_mat)[:self.data_out.shape[0]]+ np.dot(der_output, self.output_weight_mat)[:self.data_out.shape[0]]+ np.dot(der_input_val, self.input_weight_mat)[:self.data_out.shape[0]]+ np.dot(der_forget_val, self.forget_weight_mat)[:self.data_out.shape[0]]

        return del_forget_val, del_input_val, del_activation_val, del_output_val, del_prev_cell_state, del_prev_hidden_state

    def update_weight_mat(self, del_forget_val, del_input_val, del_activation_val, del_output_val):
        self.update_forget_weight_mat = 0.85 * self.update_forget_weight_mat + 0.15 * del_forget_val * del_forget_val
        self.update_input_weight_mat = 0.85 * self.update_input_weight_mat + 0.15 * del_input_val * del_input_val
        self.update_activation_weight_mat = 0.85 * self.update_activation_weight_mat + 0.15 * del_activation_val * del_activation_val
        self.update_output_weight_mat = 0.85 * self.update_output_weight_mat + 0.15 * del_output_val * del_output_val

        self.forget_weight_mat -= self.alpha/np.sqrt(np.power(10.0, -8) + self.update_forget_weight_mat) * del_forget_val
        self.input_weight_mat -= self.alpha/np.sqrt(np.power(10.0, -8) + self.update_input_weight_mat) * del_input_val
        self.activation_weight_mat -= self.alpha/np.sqrt(np.power(10.0, -8) + self.update_activation_weight_mat) * del_activation_val
        self.output_weight_mat -= self.alpha/np.sqrt(np.power(10.0, -8) + self.update_output_weight_mat) * del_output_val
