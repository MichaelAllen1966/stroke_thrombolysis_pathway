import pandas as pd

lr = pd.read_csv('logistic_regression/roc_summary_mean.csv')
forest = pd.read_csv('forest/roc_summary_mean.csv')
svm_linear = pd.read_csv('svm_linear/roc_summary_mean.csv')
svm_rbf = pd.read_csv('svm_rbf/roc_summary_mean.csv')
nn = pd.read_csv('nn/roc_summary_mean.csv')


"""Plot receiver operator curve."""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




# Define the size of the overall figure
fig = plt.figure(figsize=(5,5)) 

# Create subplot 1

ax1 = fig.add_subplot(111) # Grid of 2x2, this is suplot 1

# Plot mran results

x1 = lr['false_positive_rate']
y1 = lr['true_positive_rate']

x2 = forest['false_positive_rate']
y2 = forest['true_positive_rate']

ax1.plot(x2, y2, 
         color = 'k', 
         linestyle = '--',
         label = 'Random forest')


ax1.set_xlabel('False positive rate')
ax1.set_ylabel('True positive rate rate')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.grid(True, which='both')
ax1.text(0.65, 0.03, 'AUC = 0.889', backgroundcolor = '1.0', fontsize=11) 
#ax1.legend()

plt.savefig('forest_roc.png',dpi=300)
plt.show()
    
