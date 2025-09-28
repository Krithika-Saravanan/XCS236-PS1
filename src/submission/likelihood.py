import torch
import torch.nn as nn
import sys


def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## 
        ## The problem asks us for (positive) log likelihood, which would be equivalent to the negative of a negative log likelihood. 
        ## Cross Entropy Loss is equivalent applying LogSoftmax on an input, followed by NLLLoss. Use reduction 'sum'.
        ## 
        ## Hint: Implementation should only takes 3~7 lines of code.
        
        ### START CODE HERE ###
        # Get logits from model
        logits, _ = model(text)  

        # Shift logits and targets to ignore first token
        logits = logits[:, :-1, :]      # predict next token
        targets = text[:, 1:]           # ignore first token

        # Compute log softmax
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Gather log probabilities of the target tokens
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        #Sum  over sequence to get total log likelihood
        total_log_likelihood = token_log_probs.sum()

        return total_log_likelihood.item()
        ### END CODE HERE ###
        raise NotImplementedError
