import numpy as np
import matplotlib.pyplot as plt

#load history from file
history = np.loadtxt('new_best_history_vals.txt')  
hist_acc = history[:,0]
hist_val_acc = history[:,1]
hist_loss = history[:,2]
hist_val_loss = history[:,3]

#Plot training/testing history for model
# summarize history for accuracy
fig, ax = plt.subplots(figsize=(10,5))
plt.subplot(121)
plt.plot(hist_acc*100.)
plt.plot(hist_val_acc*100.)
plt.title('Model Accuracy',fontsize=18)
plt.ylabel('Accuracy (%)',fontsize=15)
plt.xlabel('Epoch',fontsize=15)
plt.legend(['Training', 'Testing'], loc='lower right')
plt.subplot(122)
# summarize history for loss
plt.plot(hist_loss)
plt.plot(hist_val_loss)
plt.title('Model Loss',fontsize=18)
plt.ylabel('Loss',fontsize=15)
plt.xlabel('Epoch',fontsize=15)
plt.legend(['Training', 'Testing'], loc='upper right')
plt.tight_layout()
plt.show()

