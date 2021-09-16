import subprocess, os, json
import numpy as np
import random

#for give model path resturs model description (attributes,attribute_category,attribute_values )
def get_model_description(model):
#     cmd_desc_base = 'java -jar ./lib/DEXxSearch-1.1-dev.jar STRUCTURE models/'
    cmd_desc_base = 'java -jar ./lib/DEXxSearch-1.2-dev.jar STRUCTURE models/'

    cmd_description = cmd_desc_base+model
    s = subprocess.check_output(cmd_description,shell=True)
    attributes = []
    attribute_values=[]
    # windows escapes newline with -5
    model_description = json.loads(str(s)[2:-3])
    for k in sorted(model_description.keys()):
        attributes.append(k)

        atr_alternatives_category = []
        atr_alternatives_values= []
        for i in range(len(model_description[k])):
            atr_alternatives_category.append(model_description[k][i]['category'])
            atr_alternatives_values.append(model_description[k][i]['value'])

        attribute_values.append(atr_alternatives_values)
    return attribute_values 

#obtains random instances to work with
def generate_random_alternatives(dataset, n):
    # return dataset[random.choices(range(len(dataset)), k=n)
    #getting the last n as "random"
    return dataset[len(dataset)-(n+1):]