import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.3f' % x)

labels = pd.read_csv('../data/train_labels.csv')
df = pd.read_csv('../data/train.csv', index_col =0)
df = df.merge(labels, on = 'sequence', how = 'left')

columns = [c for c in df.columns if c not in ['sequence', 'subject', 'step', 'state']]

print(df[columns].describe())

train_id = pd.read_csv('../data/train_id.csv')
val_id = pd.read_csv('../data/val_id.csv')
print("Training length: ", len(train_id))
print("Val length:", len(val_id))

ft = 22
# state
# plt.rcParams["figure.figsize"] = (10,8)
# sns.countplot( x = 'state', data = df)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.xlabel('State', fontsize=ft)
# plt.ylabel('Count', fontsize=ft)
# plt.show()

# for c in columns[:1]:
#     sns.displot(df['sensor_00'], kde = True, bins = 1000)
#     plt.xlim(-20,20)
#     plt.savefig(f'./outdir/{c}.png')
#     plt.show()

# heatmap
# columns.append('state')
# corr = df[columns].corr()
# print(corr)
# sns.heatmap(corr, annot=True, fmt=".2f")
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.show()

# kde
# def target_kde_plot(df, columns, target, ncol=4, figsize=(24, 12)):
#     nrow = round(len(columns) / ncol)
#     fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
#     col, row = 0, 0
#     for col_name in columns:
#         if nrow <= 1:
#             sns.kdeplot(data=df, x=col_name, hue=target, ax=axes[col])
#             col += 1
#         else:
#             sns.kdeplot(data=df, x=col_name, hue=target, ax=axes[row][col])
#             row += 1
#             if row >= nrow:
#                 col += 1
#                 row = 0

# columns.remove('sensor_00')
# target_kde_plot(df, columns=columns, target = 'state')
# plt.show()