import pandas as pd
#this program intent is to set index to our dataframe
#method 1
a=pd.read_excel("friendrrr.xlsx")
df=pd.Dataframe(a)
df.index=["first","second","third","fourth","fifth"]
df.set_index("Name",inplace=True)
df.iloc[1]
#method 2
b=pd.read_excel("friendrrr.xlsx",index_col="Name")
B.sort_index(inplace=True)
B.sort_index(ascending=False)

