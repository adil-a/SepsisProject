import fileio
from matplotlib import pyplot as plt
import pandas as pd

ONE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
trainDF = fileio.pklOpener(ONE_DIR_UP + '/trainSetDForiginal.pkl')
cols = list(trainDF.columns)[:-2]
onerows = trainDF.loc[trainDF['SepsisLabel'] == 1]
zerorows = trainDF.loc[trainDF['SepsisLabel'] == 0]

filenumber = 0
dictionary = {}

for i in range(len(cols)):
    if i % 10 == 0:
        filenumber += 1
        fig, [[ax1, ax2, ax3, ax4, ax5], [ax6, ax7, ax8, ax9, ax10]] = \
            plt.subplots(2, 5, figsize=(10, 5))
        dictionary = {'ax0': ax1, 'ax1': ax2, 'ax2': ax3, 'ax3': ax4, \
            'ax4': ax5, 'ax5': ax6, 'ax6': ax7, 'ax7': ax8, 
            'ax8': ax9, 'ax9': ax10}
    temp = dictionary['ax' + str(i % 10)]
    onerows.hist(column=[cols[i]], ax=temp, alpha=0.5, density=True, label='1')
    zerorows.hist(column=[cols[i]], ax=temp, alpha=0.5, density=True, label='0')
    temp.legend(loc='upper right')
    if (i + 1) % 10 == 0:
        fig.savefig('plot' + str(filenumber))