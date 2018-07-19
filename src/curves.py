import matplotlib.pyplot as plt

def generate(modelDict, fname) :
    # plotting the loss and metric for the model
    ax1 = plt.subplot(2,1,1)
    plt.plot(modelDict['mean_absolute_error'], linestyle='--', marker='o', color='b')
    plt.plot(modelDict['val_mean_absolute_error'], linestyle='--', marker='o', color='g')
    level1 = modelDict['mean_absolute_error'][0]
    level2 = modelDict['mean_absolute_error'][0] - 0.01
    i = 0
    for e in modelDict['mean_absolute_error']:
        if i%5==4 :
            e = round(e,4)
            ax1.annotate('('+str(i)+','+str(e)+')', xy=(i,level1), textcoords='data',color='b')
        i = i+1
    i = 0
    for e in modelDict['val_mean_absolute_error']:
        if i%5==4 :
            e = round(e,4)
            ax1.annotate('('+str(i)+','+str(e)+')', xy=(i,level2), textcoords='data',color='g')
        i = i+1
    plt.title('MAE | mean absolute error')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower left')

    ax2 = plt.subplot(2,1,2)
    plt.plot(modelDict['loss'], linestyle='--', marker='o', color='b')
    plt.plot(modelDict['val_loss'], linestyle='--', marker='o', color='g')
    level1 = modelDict['loss'][0]
    level2 = modelDict['loss'][0] - 0.01
    i = 0
    for e in modelDict['loss']:
        if i%5==4 :
            e = round(e,4)
            ax2.annotate('('+str(i)+','+str(e)+')', xy=(i,level1), textcoords='data',color='b')
        i = i+1
    i = 0
    for e in modelDict['val_loss']:
        if i%5==4 :
            e = round(e,4)
            ax2.annotate('('+str(i)+','+str(e)+')', xy=(i,level2), textcoords='data',color='g')
        i = i+1
    plt.title('MSE | mean squared error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower left')

    # exporting the plot
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((16,9), forward=False)
    plt.savefig('results/'+fname+'.png')
    plt.close()
