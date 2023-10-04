# import torch
# import torch.nn as nn
# # Target are to be padded
# T = 50      # Input sequence length
# C = 20      # Number of classes (including blank)
# N = 16      # Batch size
# S = 30      # Target sequence length of longest target in batch (padding length)
# S_min = 10  # Minimum target length, for demonstration purposes
# # Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(input, target, input_lengths, target_lengths)
# print(target)

import torch
import torch.nn as nn

# Define the inputs and targets
log_probs = torch.tensor([[[0.1, 0.3, 0.6], [0.2, 0.4, 0.4], [0.3, 0.5, 0.2]],
                          [[0.4, 0.5, 0.1], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5]]], dtype=torch.float32)  # (T, N, C)
targets = torch.tensor([[1, 2], [2, 0]], dtype=torch.int32)  # (N, S)
input_lengths = torch.tensor([3, 3], dtype=torch.int32)  # (N,)
target_lengths = torch.tensor([2, 2], dtype=torch.int32)  # (N,)

# # Create the CTC loss instance
# ctc_loss = nn.CTCLoss()

# # Compute the loss
# loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(log_probs.shape)




# from config import ModelConfigs

# configs = ModelConfigs()
# print(configs.learning_rate)
# def accuracies(actual_labels,predicted_labels):
#     """
#     Takes a List of Actual Outputs, predicted Outputs and returns their accuracy and letter accuracy across
#     all the labels in the list
#     """
#     accuracy=0
#     letter_acc=0
#     letter_cnt=0
#     count=0
#     for i in range(len(actual_labels)):
#         predicted_output=predicted_labels[i]
#         actual_output=actual_labels[i]
#         count+=1
#         for j in range(min(len(predicted_output),len(actual_output))):
#             if predicted_output[j]==actual_output[j]:
#                 letter_acc+=1
#         letter_cnt+=max(len(predicted_output),len(actual_output))
#         if actual_output==predicted_output:
#             accuracy+=1
#     final_accuracy=np.round((accuracy/len(actual_labels))*100,2)
#     final_letter_acc=np.round((letter_acc/letter_cnt)*100,2)
#     return final_accuracy,final_letter_acc
            
# def show_accuracy_metrics(self,num_batches, is_train):
#         """
#         Calculates the accuracy and letter accuracy for each batch of inputs, 
#         and prints the avarage accuracy and letter accuracy across all the batches
#         """
#         accuracy=0
#         letter_accuracy=0
#         batches_cnt=num_batches
#         while batches_cnt>0:
#             word_batch = next(self.text_img_gen)[0]   #Gets the next batch from the Data generator
#             decoded_res = decode_batch(self.test_func,word_batch['img_input'])
#             actual_res=word_batch['source_str']
#             acc,let_acc=accuracies(actual_res,decoded_res,self.is_train)
#             accuracy+=acc
#             letter_accuracy+=let_acc
#             batches_cnt-=1
#         accuracy=accuracy/num_batches
#         letter_accuracy=letter_accuracy/num_batches
#         if is_train:
#             print("Train Average Accuracy of "+str(num_batches)+" Batches: ",np.round(accuracy,2)," %")
#             print("Train Average Letter Accuracy of "+str(num_batches)+" Batches: ",np.round(letter_accuracy,2)," %")
#         else:
#             print("Validation Average Accuracy of "+str(num_batches)+" Batches: ",np.round(accuracy,2)," %")
#             print("Validation Average Letter Accuracy of "+str(num_batches)+" Batches: ",np.round(letter_accuracy,2)," %")


