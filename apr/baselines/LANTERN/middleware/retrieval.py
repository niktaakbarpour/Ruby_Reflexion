import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Adjust if needed
sys.path.append(project_root)
import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json
import jsonlines
from evaluator.get_result import estimate_pass_at_k
from middleware.history import get_historical_chain
import time
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


LANG_CLUSTER_TO_LANG_COMPILER = {
    "C": "GNU C11",
    "C#": "Mono C#",
    "C++": "GNU C++17",
    "Go": "Go",
    "Java": "Java 17",
    "Javascript": "Node.js",
    "Kotlin": "Kotlin 1.4",
    "PHP": "PHP",
    "Python": "PyPy 3",
    "Ruby": "Ruby 3",
    "Rust": "Rust 2018",
}

def init_vec_db(base_dir, local_dataset_path):
    '''
    encode all bugs
    '''
    vec_db_dir = os.path.join(base_dir, 'vec_db')
    os.makedirs(vec_db_dir, exist_ok=True)
    save_path = os.path.join(vec_db_dir, 'vec_db.json')
    if os.path.exists(save_path):
        return
    apr_dataset = datasets.load_from_disk(local_dataset_path)
    categorical_features = []
    tag_features = []
    bug_uids = []

    initial_ids = set()
    eval_dir = os.path.join(base_dir, "eval_apr_val_execeval")
    for filename in os.listdir(eval_dir):
        file_path = os.path.join(eval_dir, filename)
        with jsonlines.open(file_path) as jrp:
            for sample in jrp:
                initial_ids.add(sample["source_data"]["bug_code_uid"])
    
    for bug in apr_dataset:
        if bug['bug_code_uid'] in initial_ids:
            bug_uids.append(bug['bug_code_uid'])
            # Categorical features
            categorical_feat = [
                bug['difficulty'],
                bug['prob_desc_time_limit'],
                bug['prob_desc_memory_limit'],
                bug['bug_exec_outcome'],
                bug['lang_cluster']
            ]
            categorical_features.append(categorical_feat)
            # Tags feature
            tag_features.append(bug['tags'])
    categorical_features = np.array(categorical_features)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(categorical_features)

    # Create binary vectors for tags
    # First, get all unique tags
    all_tags = set()
    for tags in tag_features:
        all_tags.update(tags)
    all_tags = sorted(list(all_tags))

    # Create tag vectors
    tag_vectors = np.zeros((len(tag_features), len(all_tags)))
    for i, tags in enumerate(tag_features):
        for tag in tags:
            tag_vectors[i, all_tags.index(tag)] = 1
    final_vectors = np.hstack([encoded_categorical, tag_vectors])

    encodings_dict = {uid: vec.tolist() for uid, vec in zip(bug_uids, final_vectors)}

    # Create a dictionary to store feature information
    feature_info = {
        'categorical_features': encoder.get_feature_names_out(
            ['difficulty', 'time_limit', 'memory_limit', 'exec_outcome', 'lang_cluster']
        ).tolist(),
        'tag_features': all_tags
    }
    with open(save_path, 'w') as f:
        json.dump({
            'encodings': encodings_dict,
            'feature_info': feature_info
        }, f)


def init_cos_similarity(base_dir):
    cos_path = os.path.join(base_dir, 'vec_db/cos.json')
    vec_path = os.path.join(base_dir, "vec_db/vec_db.json")
    if not os.path.exists(cos_path):
        # Load vector database
        with open(vec_path, 'r') as f:
            vec_db = json.load(f)
        
        # Get all encodings
        encodings = vec_db['encodings']
        
        # Convert to list of vectors and keep track of ids
        vectors = []
        ids = []
        for code_id, vector in encodings.items():
            vectors.append(vector)
            ids.append(code_id)
            
        # Convert to numpy array
        vectors = np.array(vectors)
        
        # Calculate cosine similarity matrix 
        cos_sim = cosine_similarity(vectors)
        
        # Create nested dictionary mapping ids to similarity scores
        cos_dict = {}
        for i in range(len(ids)):
            id1 = ids[i]
            cos_dict[id1] = {}
            for j in range(len(ids)):
                id2 = ids[j]
                if i != j:  # Skip diagonal (self-similarity)
                    cos_dict[id1][id2] = float(cos_sim[i][j])
        
        # Save to json file
        os.makedirs(os.path.dirname(cos_path), exist_ok=True)
        with open(cos_path, 'w') as f:
            json.dump(cos_dict, f)
            
    return cos_path


def sanitize_code(code):
    prefixes = ["csharp", "cpp", "go", "javascript", "kotlin", "php", "python", "ruby", "rust", "c", "java", "json"]
    
    # Find the first occurrence of ```
    start_idx = code.find("```")
    if start_idx == -1:
        return code  # No code block found
    
    # Extract everything from the first ``` onwards
    code_block = code[start_idx:]
    
    # Remove the opening ```
    code_block = code_block[3:]
    
    # Check if it starts with a language prefix and remove it
    for prefix in prefixes:
        if code_block.startswith(prefix):
            code_block = code_block[len(prefix):]
            break
    
    # Find the closing ```
    end_idx = code_block.find("```")
    if end_idx != -1:
        code_block = code_block[:end_idx]
    
    code_block = code_block.strip()


    if not code_block.endswith('}'):
        code_block += '}'
    
    return code_block



def build_target_db(base_dir, it):
    print("IN build_target_db")
    """
    Build after decide, before translation. 
    Save as {bug_code_uid: target language}
    """
    dec_dir = os.path.join(base_dir, f'iter_{it}/decide')
    dec_path = os.path.join(base_dir, f'iter_{it}/decision.json')
    if os.path.exists(dec_path):
        with open(dec_path, "r") as dec_f:
            target_db = json.load(dec_f)
    else:
        target_db = dict()

    for filename in os.listdir(dec_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(dec_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"❌ [data: {data}")

            code_id = data["source_data"]["bug_code_uid"]
            print(f"❌ [code_id: {code_id}")

            if code_id not in target_db:
                # Extract response content
                res = data["oai_response"]["choices"][0]["message"]["content"]
                print(f"❌ [res: {res}")
                # Get sanitized code and parse as JSON
                code = sanitize_code(res)
                print(f"❌ [code: {code}")


                code = code.replace("\n", "") # claude 3.5 sometimes has this problem...
                print(f"❌ [code2: {code}")

                try:
                    code_json = json.loads(code)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed for {code_id}: {e}")
                    print(f"Attempting to fix JSON...")
                    
                    # Try to fix common issues
                    import re
                    fixed_code = re.sub(r'(\d+)\s+([a-zA-Z]+)', r'"\1 \2"', code)
                    
                    try:
                        code_json = json.loads(fixed_code)
                        print(f"Successfully fixed JSON for {code_id}")
                        target_lang = code_json["Target Language"]
                    except json.JSONDecodeError:
                        print(f"Could not fix JSON for {code_id}, defaulting to C...")
                        with open(os.path.join(base_dir, f"iter_{it}", "bad_json.txt"), "a", encoding="utf-8") as op:
                            op.write(f"Code ID: {code_id}\n")
                            op.write(fixed_code + "\n")
                            op.write("-" * 60 + "\n")
                        target_lang = "C"

                    # Store in target_db
                    target_db[code_id] = target_lang

                
                # Get target language and code ID
                target_lang = code_json["Target Language"]
                
                
                # Store in target_db
                target_db[code_id] = target_lang
    
    # Save target_db to decision.json
    with open(dec_path, 'w') as f:
        json.dump(target_db, f)
        
    return target_db



def update_pass_10(base_dir, it):
    '''
    calculate pass@10 for each fixed bug in the last iteration.
    '''
    fixed_file = os.path.join(base_dir, 'vec_db/each_pass_10.json')
    if os.path.exists(fixed_file):
        with open(fixed_file, "r") as f:
            pass_10_d = json.load(f)
    else:
        pass_10_d = dict()
    if it == 1:
        eval_dir = os.path.join(base_dir, "eval_apr_val_execeval")
    else:
        eval_dir = os.path.join(base_dir, f"iter_{it - 1}/eval")
    tmp = dict()
    for filename in os.listdir(eval_dir):
        file_path = os.path.join(eval_dir, filename)
        with jsonlines.open(file_path) as jrp:
            for data in jrp:
                uid = data["source_data"]["bug_code_uid"]
                if uid not in tmp:
                    tmp[uid] = dict()
                    tmp[uid]["total"] = 0
                    tmp[uid]["correct"] = 0
                ut_res = data['unit_test_results'][0]
                if all(x["exec_outcome"] == "PASSED" for x in ut_res):
                    tmp[uid]["correct"] += 1
                tmp[uid]["total"] += 1
    for uid in tmp.keys():
        total = np.array([tmp[uid]["total"]])
        correct = np.array([tmp[uid]["correct"]])
        pass_10_d[uid] = round(estimate_pass_at_k(total, correct, 1).mean(), 4)
    with open(fixed_file, "w") as f:
        json.dump(pass_10_d, f)


def retrieve_pass_10(base_dir, uid):
    eval_file = os.path.join(base_dir, 'vec_db/each_pass_10.json')
    with open(eval_file, 'r') as f:
        data = json.load(f)
    return data[uid]


def retrieve_base(base_dir, it, bug_code_uid, top_k, initial_unfixed_ids, bug_properties, cos, print_mode=False):
    '''
    Retrieve top-k most similar bugs in the initial repair based on cosine similarity and show their properties.

    Args:
        bug_code_uid (str): The ID of the query bug
        top_k (int): Number of similar bugs to retrieve
        encodings_path (str): Path to the JSON file containing encodings
        dataset_path (str): Path to the original dataset
    
    Returns:
        tuple: (query_df, similar_bugs_df) Pandas DataFrames containing query and similar bugs info
    '''

    encodings_path = os.path.join(base_dir, "vec_db/vec_db.json")
    # Load encodings from JSON
    with open(encodings_path, 'r') as f:
        data = json.load(f)
    
    encodings_dict = data['encodings']

    
    if bug_code_uid not in encodings_dict:
        raise KeyError(f"Query uid {bug_code_uid} is not encoded in vec_db.")
    
    # Get the query vector and query bug's language cluster
    query_vector = np.array(encodings_dict[bug_code_uid])
    query_lang_cluster = bug_properties[bug_code_uid]['lang_cluster']
    trans_history = get_historical_chain(base_dir, bug_code_uid, it, 'target_lang')
    
    # Calculate cosine similarity with bugs from different language clusters
    similarities = []
    similarities_tight = []
    similarities_relax = []
    for uid, vector in encodings_dict.items():
        if (bug_properties[uid]['lang_cluster'] != query_lang_cluster and # Skip same language cluster
            uid not in initial_unfixed_ids and # Skip the failed bugs in the initial repair
            bug_properties[uid]['lang_cluster'] not in trans_history):  
            similarity = cos[bug_code_uid][uid]
            similarities_tight.append((uid, similarity))
        if (bug_properties[uid]['lang_cluster'] != query_lang_cluster and # Skip same language cluster
            uid not in initial_unfixed_ids):  # Skip the failed bugs in the initial repair
            similarity = cos[bug_code_uid][uid]
            similarities_relax.append((uid, similarity))
    if len(similarities_tight) < top_k:
        similarities = similarities_tight + similarities_relax
    else:
        similarities = similarities_tight

    # Sort by similarity score in descending order and get top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_results = similarities[:top_k]

    # Create DataFrame for similar bugs
    similar_bugs_data = []
    for i, item in enumerate(top_k_results):
        uid, similarity = item
        bug_data = bug_properties[uid].copy()
        bug_data['similarity'] = round(similarity, 4)
        bug_data["pass@10"] = retrieve_pass_10(base_dir, uid)
        similar_bugs_data.append(bug_data)
    
    if print_mode:
        # Create DataFrame for query bug
        query_df = pd.DataFrame([bug_properties[bug_code_uid]])
        
        similar_bugs_df = pd.DataFrame(similar_bugs_data)
        
        # Reorder columns for better presentation
        column_order = ['bug_code_uid', 'similarity', 'difficulty', 'bug_exec_outcome', 
                    'lang_cluster', 'prob_desc_time_limit', 'prob_desc_memory_limit', 'tags', "pass@10"]
        similar_bugs_df = similar_bugs_df.reindex(columns=[col for col in column_order if col in similar_bugs_df.columns])
    else:
        query_df = bug_properties[bug_code_uid]
        
        trans_history_str = to_str(trans_history)
        query_df["translation_history"] = trans_history_str
        similar_bugs_df = similar_bugs_data
    
    return query_df, similar_bugs_df


def to_str(hist):
    """
    transform list ['a', 'b', 'c'] to string form '[a, b, c]'
    """
    hist_str = '[' + ', '.join(hist) + ']'
    return hist_str


def retrieve_trans(base_dir, it, bug_code_uid, top_k, trans_fixed_ids, bug_properties, cos, print_mode):
    similarities = []
    similarities_tight = []
    similarities_relax = []
    attempted = get_historical_chain(base_dir, bug_code_uid, it, 'target_lang')
    query_lang_cluster = bug_properties[bug_code_uid]['lang_cluster']

    for uid in trans_fixed_ids:
        similarity = cos[bug_code_uid][uid]
        trans_history = get_historical_chain(base_dir, uid, it, 'target_lang')
        if trans_history[-1] not in attempted and bug_properties[uid]['lang_cluster'] == query_lang_cluster:
            similarities_tight.append((uid, similarity))
        else:
            similarities_relax.append((uid, similarity))
    if len(similarities_tight) < top_k:
        similarities = similarities_tight + similarities_relax
    else:
        similarities = similarities_tight
    # print(len(similarities_tight), len(similarities_relax), len(similarities))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_results = similarities[:top_k]
    similar_bugs_data = []
    for uid, similarity in top_k_results:
        bug_data = bug_properties[uid].copy()
        bug_data['similarity'] = round(similarity, 4)
        trans_history = get_historical_chain(base_dir, uid, it, 'target_lang')
        trans_history_str = to_str(trans_history)
        bug_data['translation_history'] = trans_history_str
        bug_data['successful_language'] = trans_history[-1]
        success_it = len(trans_history)
        bug_data['pass@10'] = retrieve_pass_10(base_dir, uid)
        similar_bugs_data.append(bug_data)

    if print_mode:
        similar_bugs_df = pd.DataFrame(similar_bugs_data)
        column_order = [
            'bug_code_uid',
            'similarity',
            'difficulty',
            'bug_exec_outcome',
            'lang_cluster',
            'prob_desc_time_limit',
            'prob_desc_memory_limit',
            'tags',
            'translation_history',
            'successful_language',
            'pass@10']
        similar_bugs_df = similar_bugs_df.reindex(columns=[col for col in column_order if col in similar_bugs_df.columns])
    else:
        similar_bugs_df = similar_bugs_data
    return similar_bugs_df


def prepare_db(base_dir, apr_dataset):
    """
    return bug properties and cosine data
    """
    bug_properties = dict()
    for bug in apr_dataset:
        uid = bug['bug_code_uid']
        bug_properties[uid] = {
            'bug_code_uid': uid,
            'difficulty': bug['difficulty'],
            'bug_exec_outcome': bug['bug_exec_outcome'],
            'lang_cluster': bug['lang_cluster'],
            'tags': bug['tags'],
            'prob_desc_time_limit': bug['prob_desc_time_limit'],
            'prob_desc_memory_limit': bug['prob_desc_memory_limit']
        }
    cos_path = os.path.join(base_dir, "vec_db/cos.json")
    with open(cos_path, "r") as cos_f:
        cos = json.load(cos_f)
    return bug_properties, cos


def process_history(history_df):
    print("DEBUG: history_df is", history_df)
    if not history_df:
        return []  # Safe fallback

    if 'translation_history' in history_df[0]:
        headers = [
            'Bug ID',
            'Similarity',
            'Problem Difficulty',
            'Bug Language',
            'Problem Tags',
            'Execution Outcome',
            'Problem Time Limit',
            'Problem Memory Limit',
            'Attempted Languages',
            'Successful Language',
            'Pass@10']
    else:
        headers = [
            'Bug ID',
            'Similarity',
            'Problem Difficulty',
            'Bug Language',
            'Problem Tags',
            'Execution Outcome',
            'Problem Time Limit',
            'Problem Memory Limit',
            'Pass@10']
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator_line = '| ' + ' | '.join(['-' * len(header) for header in headers]) + ' |'
    table_lines = [
        header_line,
        separator_line]
    for i, bug_data in enumerate(history_df):
        bug_id = i + 1
        similarity = bug_data.get('similarity', '')
        difficulty = bug_data.get('difficulty', '')
        bug_language = bug_data.get('lang_cluster', '')
        problem_tags = to_str(bug_data.get('tags', ''))
        execution_outcome = bug_data.get('bug_exec_outcome', '')
        time_lmt = bug_data.get('prob_desc_time_limit', '')
        mem_lmt = bug_data.get('prob_desc_memory_limit', '')
        pass_10 = bug_data.get('pass@10', '')
        if 'translation_history' in history_df[0]:
            trans_hist = bug_data.get('translation_history', '')
            success_lang = bug_data.get('successful_language', '')
            row = f"| {bug_id} | {similarity} | {difficulty} | {bug_language} | {problem_tags} | {execution_outcome} | {time_lmt} | {mem_lmt} | {trans_hist} | {success_lang} | {pass_10} |"
        else:
            row = f"| {bug_id} | {similarity} | {difficulty} | {bug_language} | {problem_tags} | {execution_outcome} | {time_lmt} | {mem_lmt} | {pass_10} |"
        table_lines.append(row)
    output = '\n'.join(table_lines)
    return output


def process_df(query_df, similar_bugs_df, similar_bugs_t_df, print_mode=False, nohist=False):
    if print_mode:
        return (query_df, similar_bugs_df, similar_bugs_t_df)
    bug_retrieval = dict()
    if nohist:
        bug_retrieval['lang'] = query_df['lang_cluster']
    else:
        bug_retrieval['bug_info'] = f"- Bug Language: {query_df['lang_cluster']}\n- Problem Tag: {query_df['tags']}\n- Problem Difficulty: {query_df['difficulty']}\n- Execution Outcome: {query_df['bug_exec_outcome']}\n- Problem Time Limit: {query_df['prob_desc_time_limit']}\n- Problem Memory Limit: {query_df['prob_desc_memory_limit']}"
    repair_hist = ""
    if similar_bugs_df is not None and len(similar_bugs_df) > 0:
        repair_hist = process_history(similar_bugs_df)
    else:
        print("WARNING: No similar_bugs_df data available, skipping repair history.")
    trans_repair_hist = '' if similar_bugs_t_df is None else process_history(similar_bugs_t_df)
    
    print(f"repair_hist: {repair_hist}")
    print(f"trans_repair_hist: {trans_repair_hist}")

    if isinstance(repair_hist, list):
        repair_hist = "\n".join(repair_hist)

    if isinstance(trans_repair_hist, list):
        trans_repair_hist = "\n".join(trans_repair_hist)

    bug_retrieval['history'] = repair_hist if trans_repair_hist == '' else repair_hist + '\n\n' + 'Historical Translation-Repair Data:\n' + trans_repair_hist
    bug_retrieval['scope'] = '[C, C#, C++, Go, Java, Javascript, Kotlin, PHP, Python, Ruby, Rust]'
    bug_retrieval['attempted'] = query_df['translation_history']
    return bug_retrieval


def retrieve(base_dir, it, bug_code_uid, top_k, bug_properties, cos, print_mode=False, nohist=False):
    iter1_dir = os.path.join(base_dir, 'iter_1')
    initial_unfixed_file = os.path.join(iter1_dir, 'unfixed.json')
    with open(initial_unfixed_file, 'r') as uf1:
        initial_unfixed_ids = set(json.load(uf1).keys())
    query_df, similar_bugs_df = retrieve_base(base_dir, it, bug_code_uid, top_k, initial_unfixed_ids, bug_properties, cos, print_mode)


    if it > 1:
        iter_dir = os.path.join(base_dir, f'iter_{it}')
        it_unfixed_file = os.path.join(iter_dir, 'unfixed.json')
        with open(it_unfixed_file, 'r') as uf2:
            it_unfixed_ids = set(json.load(uf2).keys())    
        trans_fixed_ids = initial_unfixed_ids - it_unfixed_ids

        similar_bugs_t_df = retrieve_trans(base_dir, it, bug_code_uid, top_k, trans_fixed_ids, bug_properties, cos, print_mode)

    else:
        similar_bugs_t_df = None
    return process_df(query_df, similar_bugs_df, similar_bugs_t_df, print_mode, nohist)


def print_bug_info(query_df, similar_bugs_df):
    """
    Helper function to print the bug information in pandas table format.
    
    Args:
        query_df (pd.DataFrame): DataFrame containing query bug information
        similar_bugs_df (pd.DataFrame): DataFrame containing similar bugs information
    """
    print("\nQuery Bug:")
    print("-" * 100)
    print(query_df.to_string(index=False))
    
    print("\nSimilar Bugs:")
    print("-" * 100)
    print(similar_bugs_df.to_string(index=False))

if __name__ == '__main__':
    # init_vec_db('/root/my/data/xCodeEval/evaluation/tr_vanilla/', '/root/my/data/xCodeEval/apr')
    # init_cos_similarity('/root/my/data/xCodeEval/evaluation/tr_base/')
    # build_target_db("/root/my/data/xCodeEval/evaluation/tr_cognitive/", 1)
    # update_pass_10("/root/my/data/xCodeEval/evaluation/tr_vanilla/", 1)
    # print(retrieve_pass_10("/root/my/data/xCodeEval/evaluation/tr_vanilla/", "7255f8fba60d7412cfa56fa935e68151"))
    # bug_properties, cos = prepare_db('/root/my/data/xCodeEval/evaluation/tr_cognitive/', "/root/my/data/xCodeEval/apr")
    # print(bug_properties["55b8ba194f8dc74d9d5ceb19d82bbbed"])
    # print(len(cos))
    base_dir = "/root/my/data/xCodeEval/evaluation/tr_cognitive_ht/"
    apr_dataset = datasets.load_from_disk('/root/my/data/xCodeEval/apr')
    it = 5
    bug_properties, cos = prepare_db(base_dir, apr_dataset)
    uid = "eec62c7245d86a911dee3d88ab7981c9"
    # iter1_dir = os.path.join(base_dir, 'iter_1')
    # initial_unfixed_file = os.path.join(iter1_dir, 'unfixed.json')
    # with open(initial_unfixed_file, 'r') as uf1:
    #     initial_unfixed_ids = set(json.load(uf1).keys())
    # query_df, similar_df = retrieve_base('/root/my/data/xCodeEval/evaluation/tr_vanilla/', it, "910ce59e3983002199f4ea07a18c5ab7", 15, initial_unfixed_ids, bug_properties, cos, True)
    # print_bug_info(query_df, similar_df)

    # iter_dir = os.path.join(base_dir, f'iter_{it}')
    # it_unfixed_file = os.path.join(iter_dir, 'unfixed.json')
    # with open(it_unfixed_file, 'r') as uf2:
    #     it_unfixed_ids = set(json.load(uf2).keys())    
    # trans_fixed_ids = initial_unfixed_ids - it_unfixed_ids
    # similar_bugs_t_df = retrieve_trans(base_dir, it, "910ce59e3983002199f4ea07a18c5ab7", 15, trans_fixed_ids, bug_properties, cos, True)
    # print_bug_info(query_df, similar_bugs_t_df)

    results = retrieve(base_dir, it, uid, 15, bug_properties, cos, True)
    # print(results)
    print_bug_info(results[0], results[1])
    print_bug_info(results[0], results[2])

    # apr_dataset = datasets.load_from_disk('/root/my/data/xCodeEval/apr')
    
    # results = retrieve('/root/my/data/xCodeEval/evaluation/tr_cognitive/', 11, '55b8ba194f8dc74d9d5ceb19d82bbbed', 15, bug_properties, cos)
    # print(results)
