import numpy as np
import matplotlib.pyplot as plt


avg_hist = np.loadtxt('binary_history_vals.txt')
hist_acc = avg_hist[:,0]
hist_val_acc = avg_hist[:,1]
hist_loss = avg_hist[:,2]
hist_val_loss = avg_hist[:,3]

val_loss_min = np.argmin(hist_val_loss)
print val_loss_min
print hist_val_loss[val_loss_min]
print hist_val_acc[val_loss_min]

# summarize history for accuracy
fig, ax = plt.subplots(figsize=(10,5))
plt.subplot(121)
plt.plot(hist_acc*100.)
plt.plot(hist_val_acc*100.)
plt.title('Model Accuracy',fontsize=18)
plt.ylabel('Accuracy (%)',fontsize=15)
plt.xlabel('Epoch',fontsize=15)
plt.legend(['Training', 'Testing'], loc='upper left')
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
