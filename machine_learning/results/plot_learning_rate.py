import pandas as pd

data = pd.read_csv('learning_rate_results.csv')

"""Plot receiver operator curve."""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




# Define the size of the overall figure
fig = plt.figure(figsize=(5,5)) 

# Create subplot 1

ax1 = fig.add_subplot(111) # Grid of 2x2, this is suplot 1

# Plot mran results

x = data['Training_set_size']
y1 = data['LR']

ax1.plot(x, y1, 
         color = 'k', 
         linestyle = '-',
         label = 'Logistic regression')

x = data['Training_set_size']
y2 = data['Forest']

ax1.plot(x, y2, 
         color = 'k', 
         linestyle = '--',
         label = 'Random forest')

x = data['Training_set_size']
y3 = data['SVM_lin']

ax1.plot(x, y3, 
         color = 'k', 
         linestyle = '-.',
         label = 'SVM (linear)')

x = data['Training_set_size']
y4 = data['SVM_rbf']

ax1.plot(x, y4, 
         color = 'k', 
         linestyle = ':',
         label = 'SVM (rbf)')

x = data['Training_set_size']
y5 = data['Neural']
ax1.plot(x, y5, 
         color = '0.5', 
         linestyle = '-',
         label = 'Neural network')


ax1.set_xlabel('Training set size')
ax1.set_ylabel('Accuracy')
#ax1.set_ylim(0.4, 1.0)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
ax1.grid(True, which='both')
ax1.legend()

plt.savefig('learning_rate.png',dpi=300)
plt.show()
    
