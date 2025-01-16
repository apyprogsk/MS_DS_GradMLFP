"""null_handler is a data preprocessing function that handles the null values present in our dataset. Within the
config.py file, users may select mean, mode, or delete to have the null data values filled with mean of their column,
 the mode of their column, or to delete every row with a null value."""


def null_handler(option, data):
    if option == 'mean':
        data_mod = data.fillna(data.mean())
        return data_mod
    elif option == 'mode':
        data_mod = data.fillna(data.mode().iloc[0])
        return data_mod
    elif option == 'delete':
        data_mod = data.dropna(axis=0)
        return data_mod
    else:
        print("Invalid selection for handling null data values.")