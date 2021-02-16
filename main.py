#%%
list_magic  = ['1', '2', 'Harry', 'Becky', 'The incredible Yeti']

dict_magic = {'1':'1', 
            '2':'crocks',
            '3':'rocks',
            'Harry':'Henderson',
            'Becky':'Banana'}

#Checking the keys of the dictionary
for idx, row in enumerate(list_magic):
	if row in dict_magic.keys():
		print(f'{row} key is in the dictionary!')
	else:
		print(f'{row} key is not in the dictionary.  Sad face')

print('\n\n')

#Checking the dictionary values
for idx, row in enumerate(list_magic):
	if row in dict_magic.values():
		print(f'{row} is in the values of the dict!')
	else:
		print(f'{row} value is not in the dictionary.  Sad face')

#%%
list_magic  = ['1', '2', 'Harry', 'Becky', 'The incredible Yeti']

dict_magic = {'1':['2','1','3','4','5'], 
            '2':['red', 'blue', 'purple'],
            '3': ['words', 'are', 'fun'],
            'Harry':['David','Joesephs', 'loves','The incredible Yeti'],
            'Becky':['Santerre', 'loves','pdb']}

def find_value_for(input_dict, value):    
    for k, v in input_dict.items():
        if value in v or value == v:
            print(f'{value} in dict_magic!')
            break
        else:
            print(f'{value} is not in dict_magic :(')

for idx, val in enumerate(list_magic):
    find_value_for(dict_magic, val)


# %%
