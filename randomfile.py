

import pandas as pd
import fileio
import os
import pickle

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
THIS_FOLDER2 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


# outstr1 = 'auc\n'
# outstr2 = 'n_estimators\n'
# outstr3 = 'reg_lambda\n'
# outstr4 = 'max_depth\n'
# outstr5 = 'min_child_weight\n'
# outstr6 = 'subsample\n'
# outstr7 = 'eta\n'
# outstr8 = 'gamma\n'
# outstr9 = 'colsample_bytree\n'
# for i in range(5):
#     for j in range(2):
#         dictionary = fileio.pklOpener(THIS_FOLDER + f"/training_set_batches/batch_{i + 1}{j + 1}_analysis.pkl")
#         outstr1 += str(dictionary['auc']) + '\n'
#         outstr2 += str(dictionary['config']['n_estimators']) + '\n'
#         outstr3 += str(dictionary['config']['reg_lambda']) + '\n'
#         outstr4 += str(dictionary['config']['max_depth']) + '\n'
#         outstr5 += str(dictionary['config']['min_child_weight']) + '\n'
#         outstr6 += str(dictionary['config']['subsample']) + '\n'
#         outstr7 += str(dictionary['config']['eta']) + '\n'
#         outstr8 += str(dictionary['config']['gamma']) + '\n'
#         outstr9 += str(dictionary['config']['colsample_bytree']) + '\n'

# print(outstr1, outstr2, outstr3, outstr4, outstr5, outstr6, outstr7, outstr8, outstr9)
# print(fileio.pklOpener('newTrainSet.pkl'))
print(THIS_FOLDER)