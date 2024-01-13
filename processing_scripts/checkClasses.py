# Function to split elements in a list by a separator ";"
def split_elements_by_separator(input_list, separator=";"):
    # Use a list comprehension to split each element by the separator
    split_elements = [item.split(separator) for item in input_list]
    
    return split_elements

# Example usage
input_list = ["apple;banana;cherry", "dog;cat", "red;blue;green"]
split_elements = split_elements_by_separator(input_list)

# Print the original list and the split elements
print("Original List:", input_list)
print("Split Elements:")
for item in split_elements:
    print(item)
    
print(split_elements)



def remove_before_semicolon(input_string):
    # Find the index of the first semicolon in the input string
    semicolon_index = input_string.find(';')
    
    if semicolon_index != -1:
        # If a semicolon is found, return the substring after it
        return input_string[semicolon_index + 1:]
    else:
        # If no semicolon is found, return the original string
        return input_string

# Example usage:
input_string = "Remove this part;Keep this part"
result = remove_before_semicolon(input_string)
print(result)


# Sample list of dictionaries
data = [
    {"id": 1, "label": "A"},
    {"id": 2, "label": ""},
    {"id": 3, "label": "B"},
    {"id": 4, "label": ""},
    {"id": 5, "name": "C"}  # No "label" key in this dictionary
]

# # Function to remove dictionaries with empty "label" values
# def remove_empty_label_dicts(data_list):
#     return [d for d in data_list if "label" not in d or d["label"] != ""]

# # Call the function to remove dictionaries with empty "label" values
# filtered_data = remove_empty_label_dicts(data)

# # Print the filtered data
# for item in filtered_data:
#     print(item)

for idx, item in enumerate(data):
    print(f'idx is :{idx} and the data should be : {item}')