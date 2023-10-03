import pandas as pd
df = pd.read_csv("iris_saved.csv")

import matplotlib.pyplot as plt

#Creating line plot

plt.plot(df['Sepal_Length'])
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.savefig('line_plot.png')      
plt.close()    

#Creating a histogram and tick marks

plt.hist(df['Sepal_Length'], bins=10)
plt.xticks([4,5,6,7,8])
plt.title('Historgram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('histogram_of_sepal_length.png')
plt.close()

#Creating a piechart

species_count=df['Class'].value_counts()
plt.pie(species_count, labels=species_count.index, autopct='%1.1f%%')
plt.title('Species Distribution')
plt.savefig('piechart.png')      
plt.close()    