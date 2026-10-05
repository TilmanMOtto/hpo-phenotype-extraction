import argparse
import os
import time
import logging
from tqdm import tqdm
from evaluate import GSCplus_corpus
from ee import process_text2phrases_pbert, process_text2phrases_benepar, process_text2phrases_abbr
import json

# PhenoRAG patch: ``dic_ner`` / ``tagging_text`` were module-level imports upstream. They belong to
# the rule-based (PhenoTagger) entity-extraction path, which this repo does not vendor -- importing
# them would fail before the neural path ran. Moved into run_gsc_test, their only caller.
# See PATCHES.md.


'''
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.Session(config=config) 
'''
def run_gsc_test(files,biotag_dic,nn_model,ts,disjoint,abbr_recog,doc_id,batch=False):
    from dic_ner import dic_ont            # PhenoRAG patch: lazy, see PATCHES.md
    from tagging_text import bioTag, bioTag_batch

    fin_test=open(files['testfile'],'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(files['outfile'],'w',encoding='utf-8')

    if batch:
        lines_batch = [doc_test.split('\n') for doc_test in all_test]
        # lines_batch = [lines for lines in lines_batch if lines[0] == '3185841']
        test_result_batch = bioTag_batch([lines[1] for lines in lines_batch],biotag_dic,nn_model,onlyLongest=False,abbrRecog=abbr_recog,Threshold=ts)
        for lines, test_result in zip(lines_batch, test_result_batch):
            pmid = lines[0]
            test_out.write(pmid+'\n'+lines[1]+'\n')
            for ele in test_result:
                test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
            test_out.write('\n')
        test_out.close()
    else:
        for doc_test in tqdm(all_test):
            lines=doc_test.split('\n')
            pmid = lines[0]
            if doc_id and pmid not in doc_id: # '10712204', '15DG2492':
                continue
            # test_result = []
            # for line_sub in nltk.sent_tokenize(lines[1]):
            #     test_result += bioTag(line_sub,biotag_dic,nn_model,onlyLongest=False,abbrRecog=False,Threshold=ts)
            test_result = bioTag(lines[1],biotag_dic,nn_model,onlyLongest=False,abbrRecog=abbr_recog,Threshold=ts)
            test_out.write(pmid+'\n'+lines[1]+'\n')
            for ele in test_result:
                test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
            test_out.write('\n')
        test_out.close()

    # only keep longest overlapped entities
    if disjoint:
        all_lines = []
        for line in open(files['outfile'], 'r').read().strip().split('\n\n'):
            line = line.split('\n')
            anns = [ann.split('\t') for ann in line[2:]]
            anns_new = []
            for i in range(len(anns)):
                overlap = False
                for j in range(len(anns)):
                    if i != j and int(anns[j][0]) <= int(anns[i][0]) and int(anns[i][1]) <= int(anns[j][1]) and int(anns[i][1]) - int(anns[i][0]) < int(anns[j][1]) - int(anns[j][0]):
                        overlap = True
                        break
                if not overlap:
                    anns_new.append(anns[i])
            line_new = '\n'.join(line[:2] + ['\t'.join(ann) for ann in anns_new])
            all_lines.append(line_new)
        all_lines = '\n\n'.join(all_lines)
        open(files['outfile'], 'w').write(all_lines)

    # PhenoRAG patch: upstream always ran its own evaluator here against the ground-truth TSV it read
    # the text from. The baseline runs stage a corpus with NO ground-truth lines (scoring is the
    # comparison's job, over the predictions), so this is guarded on an explicit ground-truth file.
    # See PATCHES.md.
    if files.get('goldfile'):
        GSCplus_corpus(files['outfile'],files['goldfile'],files['ontology_dict'],subtree=True)


# PhenoRAG patch: ``remove_dup`` and the benepar/conjunct cache loop were inline in
# run_gsc_test_ner. They are lifted here VERBATIM so that src/AutoPCR/extract_phrases.py can run the
# constituency parse in its own environment (autopcr_ee_venv -- upstream's transformers 4.49 /
# protobuf 6 pins, which PhenoRAG_marc_env cannot hold) and leave the two JSON files that
# run_gsc_test_ner already reuses when present. One copy of the code, so the parse run elsewhere
# is the parse upstream runs. See PATCHES.md.
def remove_dup(phrases):
    for pmid in phrases:
        new_phrases = []
        seen = set()
        for p in phrases[pmid]['phrases']:
            key = (p['start'], p['end'], p['phrase'])
            if key not in seen:
                seen.add(key)
                new_phrases.append(p)
        phrases[pmid]['phrases'] = new_phrases


def build_benepar_phrase_cache(all_test, test_path):
    phrasefile_benepar = test_path + f'/phrases_benepar.json'
    phrasefile_conjunct = test_path + f'/phrases_conjunct.json'
    all_phrases_benepar, all_phrases_conjunct = {}, {}
    for doc_test in tqdm(all_test):
        lines = doc_test.split('\n')
        text, phrases_list, index_map = process_text2phrases_benepar(lines[1])
        all_phrases_benepar[lines[0]] = {'text_new': text}
        all_phrases_benepar[lines[0]]['phrases'] = [{'start': index_map[p[0]], 'end': index_map[p[1]], 'phrase': p[2], 'phrase_nostopword': p[3]} for p in phrases_list[0]]
        all_phrases_conjunct[lines[0]] = {'text_new': text}
        all_phrases_conjunct[lines[0]]['phrases'] = [{'start': index_map[p[0]], 'end': index_map[p[1]], 'phrase': p[2], 'phrase_nostopword': p[3]} for p in phrases_list[1]]
    remove_dup(all_phrases_benepar)
    remove_dup(all_phrases_conjunct)
    json.dump(all_phrases_benepar, open(phrasefile_benepar, 'w'))
    json.dump(all_phrases_conjunct, open(phrasefile_conjunct, 'w'))
    return all_phrases_benepar, all_phrases_conjunct


def run_gsc_test_ner(files,nn_model,ts,disjoint,abbr_recog,ee,doc_id):
    fin_test=open(files['testfile'],'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(files['outfile'],'w',encoding='utf-8')
    test_out_raw=open(files['outfile'][:-4]+'.raw.tsv','w',encoding='utf-8')
    test_path = '/'.join(files['testfile'].split('/')[:-1])

    # Handling phrases
    all_phrases = {doc_test.split('\n')[0]: {'phrases': []} for doc_test in all_test}
    # PhenoRAG patch: ``remove_dup`` was defined here; it is now module-level (above), unchanged.
    def add_phrases(phrases, phrases_to_add):
        for pmid in phrases:
            locs = set((p['start'], p['end'], p['phrase']) for p in phrases[pmid]['phrases'])
            phrases[pmid]['phrases'] += [p for p in phrases_to_add[pmid]['phrases'] if (p['start'], p['end'], p['phrase']) not in locs]


    if abbr_recog:
        phrasefile = test_path + '/phrases_abbr.json'
        if os.path.exists(phrasefile):
            all_phrases_abbr = json.load(open(phrasefile, 'r'))
        else:
            all_phrases_abbr = {}
            for doc_test in tqdm(all_test):
                lines = doc_test.split('\n')
                text, phrases_list, index_map = process_text2phrases_abbr(lines[1])
                all_phrases_abbr[lines[0]] = {'text_new': text}
                all_phrases_abbr[lines[0]]['phrases'] = [{'start': index_map[p[0]], 'end': index_map[p[1]], 'phrase': p[2], 'phrase_exp': p[3], 'phrase_nostopword': p[4]} for p in phrases_list]
            remove_dup(all_phrases_abbr)
            json.dump(all_phrases_abbr, open(phrasefile, 'w'))
        add_phrases(all_phrases, all_phrases_abbr)

    phrasefile = test_path + '/phrases.json'
    if os.path.exists(phrasefile):
        all_phrases_stanza = json.load(open(phrasefile, 'r'))
    else:
        all_phrases_stanza = {}
        for doc_test in tqdm(all_test):
            lines = doc_test.split('\n')
            text, phrases_list, index_map = process_text2phrases_pbert(lines[1])
            all_phrases_stanza[lines[0]] = {'text_new': text}
            all_phrases_stanza[lines[0]]['phrases'] = [{'start': index_map[int(p.start_loc)], 'end': index_map[int(p.end_loc)], 'phrase': text[int(p.start_loc):int(p.end_loc)], 'phrase_nostopword': p.toSimpleString()} for p in phrases_list]
        remove_dup(all_phrases_stanza)
        json.dump(all_phrases_stanza, open(phrasefile, 'w'))
    add_phrases(all_phrases, all_phrases_stanza)
    
    if ee == 'neural+' or ee == 'neural++':
        phrasefile_benepar = test_path + f'/phrases_benepar.json'
        phrasefile_conjunct = test_path + f'/phrases_conjunct.json'
        if os.path.exists(phrasefile_benepar):
            all_phrases_benepar = json.load(open(phrasefile_benepar, 'r'))
            all_phrases_conjunct = json.load(open(phrasefile_conjunct, 'r'))
        else:
            # PhenoRAG patch: the loop that was inline here is build_benepar_phrase_cache below,
            # verbatim, so extract_phrases.py can run it in the parser's own environment. See PATCHES.md.
            all_phrases_benepar, all_phrases_conjunct = build_benepar_phrase_cache(all_test, test_path)
        add_phrases(all_phrases, all_phrases_benepar)
        if ee == 'neural++':
            add_phrases(all_phrases, all_phrases_conjunct)

    for doc_test in tqdm(all_test):
        lines = doc_test.split('\n')
        pmid = lines[0]
        if doc_id and pmid not in doc_id: # 'ff036564bd31cdabd6c5cf7b5ace5840'
            continue
        annotations = nn_model.predict_llm([phrase.get('phrase_exp', phrase['phrase']) for phrase in all_phrases[pmid]['phrases']], lines[1] * len(all_phrases[pmid]['phrases']))
        test_out_raw.write(pmid+'\n'+lines[1]+'\n')
        test_out.write(pmid+'\n'+lines[1]+'\n')
        
        anns = []
        for phrase, ele in zip(all_phrases[pmid]['phrases'], annotations): # annotations.split('\n')
            start, end = phrase['start'], phrase['end']
            uri, score = ele[0] # ele.split('\t')[0].split('|')
            score = min(float(score), 1.0)
            if abs(score) >= ts:
                anns.append((str(start), str(end), phrase['phrase'], uri, str(score)))
        anns = list({(ann[0], ann[1], ann[3]): ann for ann in anns}.values()) # remove duplicates in phrases
        
        for ann in anns:
            test_out_raw.write(ann[0]+'\t'+ann[1]+'\t'+ann[2]+'\t'+ann[3]+'\t'+ann[4]+'\n')
        test_out_raw.write('\n')

        # # overlap condition: inner / outer
        # anns_new = []
        # for i in range(len(anns)):
        #     overlap = False
        #     for j in range(len(anns)):
        #         if i != j:
        #             cond = int(anns[i][0]) <= int(anns[j][0]) and int(anns[j][1]) <= int(anns[i][1]) and int(anns[j][1]) - int(anns[j][0]) < int(anns[i][1]) - int(anns[i][0]) and anns[i][3] == anns[j][3]
        #             # cond = int(anns[j][0]) <= int(anns[i][0]) and int(anns[i][1]) <= int(anns[j][1]) and int(anns[i][1]) - int(anns[i][0]) < int(anns[j][1]) - int(anns[j][0]) and anns[i][3] == anns[j][3]
        #             if cond:
        #                 overlap = True
        #                 break
        #     if not overlap:
        #         anns_new.append(anns[i])
        # anns = anns_new

        # intersect condition
        anns_new = []
        for i in range(len(anns)):
            overlap = False
            for j in range(len(anns)):
                if i != j:
                    # cond = max(int(anns[i][0]), int(anns[j][0])) < min(int(anns[i][1]), int(anns[j][1])) and anns[i][3] == anns[j][3] and (float(anns[i][4]) < float(anns[j][4]) or (float(anns[i][4]) == float(anns[j][4]) and int(anns[i][0]) > int(anns[j][0])))
                    len_i, len_j = int(anns[i][1]) - int(anns[i][0]), int(anns[j][1]) - int(anns[j][0])
                    cond = max(int(anns[i][0]), int(anns[j][0])) < min(int(anns[i][1]), int(anns[j][1])) and anns[i][3] == anns[j][3] and (float(anns[i][4]) < float(anns[j][4]) or (float(anns[i][4]) == float(anns[j][4]) and (len_i > len_j or (len_i == len_j and int(anns[i][0]) > int(anns[j][0])))))
                    if cond:
                        overlap = True
                        break
            if not overlap:
                anns_new.append(anns[i])
        
        # only keep longest nested entities
        if disjoint:
            def is_word_subsequence(short, long):
                it = iter(long.lower().split())
                return all(any(w == x for x in it) for w in short.lower().split())
            anns = anns_new
            anns_new = []
            for i in range(len(anns)):
                overlap = False
                for j in range(len(anns)):
                    if i != j and int(anns[j][0]) <= int(anns[i][0]) and int(anns[i][1]) <= int(anns[j][1]) and int(anns[i][1]) - int(anns[i][0]) < int(anns[j][1]) - int(anns[j][0]) and anns[i][2].lower() in anns[j][2].lower():
                        overlap = True
                        break
                    if i != j and int(anns[j][0]) == int(anns[i][0]) and int(anns[i][1]) == int(anns[j][1]) and is_word_subsequence(anns[i][2], anns[j][2]):
                        overlap = True
                        break
                if not overlap:
                    anns_new.append(anns[i])

        for ann in anns_new:
            test_out.write(ann[0]+'\t'+ann[1]+'\t'+ann[2]+'\t'+ann[3]+'\t'+ann[4]+'\n')
        test_out.write('\n')

    test_out_raw.close()
    test_out.close()

    # PhenoRAG patch: upstream always ran its own evaluator here against the ground-truth TSV it read
    # the text from. The baseline runs stage a corpus with NO ground-truth lines (scoring is the
    # comparison's job, over the predictions), so this is guarded on an explicit ground-truth file.
    # See PATCHES.md.
    if files.get('goldfile'):
        GSCplus_corpus(files['outfile'],files['goldfile'],files['ontology_dict'],subtree=True)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--ontology_dict', help='folder of the ontology dictionary', type=str, required=True)
    parser.add_argument('--corpus', '-c', help='input corpus dataset', type=str, required=True)
    parser.add_argument('--output', '-o', help='output prediction file', type=str, required=True)
    parser.add_argument('--doc_id', help='specific docs to test', nargs='*', type=str)

    parser.add_argument('--ee', help='entity extraction method: (1) "rule": PhenoTagger-style rule-based, (2) "neural": BioNER, (3) "neural+": BioNER + benepar, (4) "neural++": BioNER + benepar + coordination decomposition', choices=['rule', 'neural', 'neural+', 'neural++'], type=str, default='neural++')
    parser.add_argument('--abbr_recog', help='whether to identify abbreviations', action='store_true')

    parser.add_argument('--ccr', help='candidate concept retrieval model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    parser.add_argument('--tau_1', help='high-confidence threshold for candidate concept retrieval', type=float, default=0.95)
    parser.add_argument('--tau_2', help='low-confidence threshold for candidate concept retrieval', type=float, default=0.85)
    parser.add_argument('--k', help='number of retrieved candidate concepts', type=int, default=5)

    parser.add_argument('--el', help='entity linking llm backend; "none" to disable', type=str, default='Qwen/Qwen3-Next-80B-A3B-Instruct')
    parser.add_argument('--api_provider', help='api provider for llm inference', choices=['openai', 'groq', 'together', 'vllm'], type=str, default='together')
    parser.add_argument('--api_key', help='api key for api provider, base url if api provider is vllm, or loaded from .env (e.g., TOGETHER_API_KEY) if not provided', type=str)
    parser.add_argument('--use_finetuning_prompt', help='simplified prompt for finetuned llm', action='store_true')
    parser.add_argument('--seed', help='seed for llm', type=int, default=0)
    parser.add_argument('--use_cache', help='whether to cache results', action='store_true')

    parser.add_argument('--only_longest', help='whether the output only keeps the longest nested entity', action='store_true')
    
    args = parser.parse_args()

    ontfiles={'dic_file':f'{args.ontology_dict}/noabb_lemma.dic',
              'word_hpo_file':f'{args.ontology_dict}/word_id_map.json',
              'hpo_word_file':f'{args.ontology_dict}/id_word_map.json'}
    biotag_dic=dic_ont(ontfiles)

    vocabfiles={'index': f'{args.ontology_dict}/main_lable.index',
                'index_to_id': f'{args.ontology_dict}/index_to_id_lable.json',
                'index_to_term': f'{args.ontology_dict}/index_to_term_lable.json',
                'id_to_concept': f'{args.ontology_dict}/id_to_concept.json',
                'firstword_to_id': f'{args.ontology_dict}/firstword_id_map.json',
                'abbr': f'{args.ontology_dict}/abbr.json'}

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # output_dir, name = os.path.split(args.output)
    # base_name, ext = os.path.splitext(name)
    # cand_path = os.path.join(output_dir, base_name + '.json')

    if args.use_finetuning_prompt:
        from nn_model3 import bioTag_SapBERT
    else:
        from nn_model2 import bioTag_SapBERT

    nn_model=bioTag_SapBERT(vocabfiles, args.ccr, args.tau_1, args.tau_2, args.k, args.el, args.api_provider, args.api_key, args.seed, args.use_cache)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.splitext(args.output)[0]+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    files={'testfile': f'../data/corpus/{args.corpus}/{args.corpus}_test_gold.tsv',
           'goldfile': f'../data/corpus/{args.corpus}/{args.corpus}_test_gold.tsv',
           'outfile': args.output,
           'ontology_dict': args.ontology_dict}
    
    start_time=time.time()
    if args.ee == 'rule':
        run_gsc_test(files,biotag_dic,nn_model,args.tau_1,args.only_longest,args.abbr_recog,args.doc_id)
    elif args.ee.startswith('neural'):
        run_gsc_test_ner(files,nn_model,args.tau_1,args.only_longest,args.abbr_recog,args.ee,args.doc_id)
    logging.info(f'{args.corpus} done: {time.time()-start_time}')