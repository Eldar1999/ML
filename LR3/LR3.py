import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from matplotlib import pyplot as plt

all_data = pd.read_csv('dataset_group.csv', header=None)
# В файле нет строки с названием столбцов, поэтому параметр header равен None.
# Интерес представляет информация об id покупателя - столбец с названием 1
# Название купленного товара хранится в столбце с названием 2

unique_id = list(set(all_data[1]))
print(len(unique_id))  # Выведем количество id

items = list(set(all_data[2]))
print(len(items))  # Выведем количество товаров

dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
            items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df)

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
print(results, end="\n\n")

results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
print(results, end="\n\n")

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length'] == 2]
print(results)
print('\nCount of result itemstes = ', len(results))

sup = np.arange(0.05, 1, 0.01)
counts = []
cols = []
for x in sup:
    a = apriori(df, min_support=x, use_colnames=True)
    counts.append(len(a))
    a['length'] = a['itemsets'].apply(lambda x: len(x))
    cols.append(
        [len(a[a['length'] == 4]), len(a[a['length'] == 3]), len(a[a['length'] == 2])])

flags = [True, True, True]
needed_sup = []
for i in range(len(cols)):
    for j in range(len(flags)):
        if flags[j]:
            if cols[i][j] == 0:
                needed_sup.append([sup[i], counts[i]])
                flags[j] = False

plt.plot(sup, counts, "-")
plt.plot([x[0] for x in needed_sup], [x[1] for x in needed_sup], "or")
plt.savefig("fig1.png")
plt.close()

results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
                new_items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit_transform(new_dataset)
new_df = pd.DataFrame(te_ary, columns=te.columns_)
print(new_df)
