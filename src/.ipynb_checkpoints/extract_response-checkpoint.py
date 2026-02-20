import pandas as pd       # for DataFrame creation, merging, and manipulation
import json               # for parsing JSON strings
import re                 # for regex cleaning of response strings

#function to extract results
#batch-level results: usage and costs
def extract_batch_level(results, batches):
    #batches
    img_names_allbatches = [img for imgs in batches.values() for img in imgs]
    batch_names = list(batches.keys())
    batches_df = pd.DataFrame({
        'batch_name':batch_names,
        'img_names': ['; '.join(imgs) for imgs in batches.values()]
    })
    print(img_names_allbatches)
    print(len(img_names_allbatches))
    
    
    #Usage and cost
    #extract usage for each batch
    usage = [batch_data["usage"] for batch_data in results.values()]
    #assign name to each batch
    usage = dict(zip(batch_names, usage))
    #convert to df using name of each element in the index column
    usage_df = pd.DataFrame.from_dict(usage, orient='index').reset_index()
    #rename index column into batch_name
    usage_df = usage_df.rename(columns={'index': 'batch_name'})
    
    #Cost
    #extract usage for each batch
    cost = [batch_data["cost"] for batch_data in results.values()]
    #assign name to each batch
    cost = dict(zip(batch_names, cost))
    #convert to df using name of each element in the index column
    cost_df = pd.DataFrame.from_dict(cost, orient='index').reset_index()
    #rename index column into batch_name
    cost_df = cost_df.rename(columns={'index': 'batch_name'})
    
    #Combine two df
    usage_cost_df = pd.merge(usage_df, cost_df, on='batch_name')
    
    #add sum
    total_usage_cost_df = pd.DataFrame([usage_cost_df.select_dtypes(include='number').sum()])
    total_usage_cost_df['batch_name'] = 'total_batch'
    usage_cost_df = pd.concat([usage_cost_df, total_usage_cost_df], ignore_index=True)

    return {"img_names_allbatches": img_names_allbatches,
            "batches_df": batches_df,
            "usage_cost_df": usage_cost_df}

#Extract response and convert to dict
def extract_response(results, img_names_allbatches, outcomes, all_names):
    #extract response (this gives a list of text elements
    responses = [batch_data["response"] for batch_data in results.values()]
    #remove "json" or "```" at the beginning of each text element
    cleaned_responses = [re.sub(r"```json|```", "", r).strip() for r in responses]
    #convert each text elements into a dict object (of length batch_size)
    parsed_dicts = [json.loads(r) for r in cleaned_responses]
    #merge all dict element into a single dict
    merged_responses = {}
    for d in parsed_dicts:
        merged_responses.update(d)
    #print(json.dumps(merged_results, indent=2))
    
    #Convert to dataframe
    #convert to a list of rows
    rows = []
    for label, info in merged_responses.items():
        # info is a dict like {"tobacco_product": "Yes", "type": "box of cigarettes"}
        row = {"img_names": label}  # first column = image label
        row.update(info)           # add all inner dict items as separate columns
        rows.append(row)
    # Convert to DataFrame
    responses_df = pd.DataFrame(rows)
    #print(responses_df)
    
    #Add true outcome
    true_responses_df = pd.DataFrame({
        'img_names': all_names,
        'tobacco_product_true': outcomes
    })
    
    #check
    a = responses_df.img_names
    b = img_names_allbatches
    print( list(set(a) - set(b))) #made up names by the model
    print( list(set(b) - set(a))) #true names that were replaced by made up ones
    
    #replace img_names column by correct names as the model may have created wrong ones
    responses_df['img_names'] = img_names_allbatches
    
    #print(true_responses_df)
    responses_df = pd.merge(responses_df, true_responses_df, on='img_names')

    return(responses_df)