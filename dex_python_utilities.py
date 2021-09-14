import subprocess, os, json
import numpy as np

#for give model path resturs model description (attributes,attribute_category,attribute_values )
def get_model_description(model):
#     cmd_desc_base = 'java -jar ./lib/DEXxSearch-1.1-dev.jar STRUCTURE models/'
    cmd_desc_base = 'java -jar ./lib/DEXxSearch-1.2-dev.jar STRUCTURE models/'

    cmd_description = cmd_desc_base+model
    s = subprocess.check_output(cmd_description,shell=True)
    attributes = []
    attribute_category=[]
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

        attribute_category.append(atr_alternatives_category)
        attribute_values.append(atr_alternatives_values)
    return attributes,attribute_category,attribute_values 

#generates random alternatives using dexi for a given class-value using java wrapper
def generate_random_alternatives(n,model,target):
    print("Generating random alternatives. n:",n,', target:',target)
#     cmd_generate_base = "java -jar ./lib/DEXxSearch-1.1-dev.jar GENERATE_INPUT_SPACE_BY_TARGET_VALUE models/"
    cmd_generate_base = "java -jar ./lib/DEXxSearch-1.2-dev.jar GENERATE_INPUT_SPACE_BY_TARGET_VALUE models/"

    constraints = ' []'
    n=" "+str(n)
    seed=" 7"
    cmd_generate = cmd_generate_base+model+target+constraints+n+seed
    s  = subprocess.check_output(cmd_generate,shell=True)
    print("Done")
    return json.loads(str(s)[2:-3])

#generates random alternatives using dexi for each calss value (low, medium high) using java wrapper
def generate_random_alternatives_DEX(n,model):
    alternatives_low = generate_random_alternatives(n,model,target=" low")
    alternatives_medium = generate_random_alternatives(n,model,target=" medium")
    alternatives_high = generate_random_alternatives(n,model,target=" high")
    
    alternatives_low.extend(alternatives_medium)
    alternatives_low.extend(alternatives_high)
    
    return alternatives_low

#convert one alternative to numeric representation
def atribute_values_to_num(alternative,attribute_values):
    alternative_numeric=[]
    values = []
    for k in sorted(alternative.keys()):
        values.append(alternative[k])
    
    for i in range(len(attribute_values)):
        num_value = list(attribute_values[i]).index(values[i])
        alternative_numeric.append(num_value)
    return np.array(alternative_numeric)


#convert one alternative to string/dict representation
def atribute_values_to_string(alternative_numeric,sorted_attribute_keys,attribute_values):
    alterenative = dict.fromkeys(sorted_attribute_keys)
    for i in range(len(attribute_values)):

        string_value = attribute_values[i][alternative_numeric[i]]
        alterenative[sorted_attribute_keys[i]]=string_value
    return alterenative

#convert list of alternative to a list of numeric representation
def alternatives_to_num(alternatives,attribute_values):
    alternatives_num = []
    for a in alternatives:
        alternatives_num.append(atribute_values_to_num(a,attribute_values))
    alternatives_num=np.stack(alternatives_num)
    return alternatives_num

#convert list of alternative to a list of string/dict representation
def alternatives_to_string(alternatives_num,sorted_attribute_keys,attribute_values):
    alternatives_string = []
    for a in alternatives_num:
        alternatives_string.append(atribute_values_to_string(a,sorted_attribute_keys,attribute_values))
    return alternatives_string

def get_exhaustive_search_results(template_string, target):
    print('Running exhaustive search...')
    alternative_path ='template.json'
    with open(alternative_path, 'w') as outfile:
        json.dump(template_string, outfile)

#     cmd_search_base =  'java -jar ./lib/DEXxSearch-1.1-dev.jar OPTIMIZE_STEPWISE_EXHAUSTIVE_SEARCH models/'
    cmd_search_base =  'java -jar ./lib/DEXxSearch-1.2-dev.jar OPTIMIZE_CASCADE_EXHAUSTIVE_SEARCH_DIRECTED_DIFF models/'
    alternative_path ='template.json'
    target =' '+target+' [] []'
    cmd = cmd_search_base+model+" "+alternative_path +target
    s = subprocess.check_output(cmd,shell=True)
    print('Done.')

    return json.loads(str(s)[2:-3])