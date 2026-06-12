# Motivation
In early experiments, I tried to ask a randomly initialised backbone to predict unique image indices of images as a pre-training strategy.
Even with intensive data augmentation, and different ways to supervise uisng the pseudo labels derived from the image indices, the backbone cannot learn semantics very well. More precisely, the knn monitoring accuracy of the backbone easily saturates at 70%, and linear probing on the backbone has even worse results. For a backbone with 70% knn accuracy, after linear probing, the accuracy dropped to 58%. This is very different from the previously reported results of established pre-training using self-supervised learning such as contrastive learning, clustering as pseudo labels, in those previous methods, linear probe always has better results than knn.

# Goal
Research why instance level supervision is not a good pre-training strategy, and why the features cannot be transferrable after linear probing. 

# General Guidelines
- Use the ground work to keep the changes minimal.
- Make google colab notebooks to run experiments.
- Whenever update the code, record the changes in a changelog file in changes folder.
- Always record design plans in md format in the folder .agents/designs