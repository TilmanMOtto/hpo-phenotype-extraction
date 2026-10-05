import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import json
import faiss
import argparse
import pandas as pd

# PhenoRAG patch: ``from dotenv import load_dotenv`` was a module-level import used only by
# get_umls_name(), the live UMLS API fallback this fork does not use (see --umls_synonyms).
# python-dotenv is not in environment.yaml. Moved into that function. See PATCHES.md.


def get_embs(all_names, tokenizer, model, bs=128, device='cuda'):   # PhenoRAG patch: device arg
    all_embs = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                        padding="max_length", 
                                        max_length=100, 
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.to(device)   # PhenoRAG patch: was v.cuda()
        cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        all_embs.append(cls_rep.cpu().detach().numpy())
    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs


def build_index(ontology_dict, tokenizer, model, device='cuda', umls_synonyms=None):
    # PhenoRAG patch: ``device`` so the index can be built on CPU, and ``umls_synonyms`` -- upstream
    # decided whether to attach UMLS synonyms by comparing the ontology_dict string to the literal
    # '../dict/HPO', and fell back to live UMLS API calls when the file was absent (using an
    # unimported ``requests``). Both are unusable here; the path is now explicit and optional.
    # See PATCHES.md.
    hpo_concepts = json.load(open(f'{ontology_dict}/obo.json', 'r'))

    # 🔹 3. 计算 embedding 并存入 Faiss
    term_to_main_label = {}  # 存储同义词 → 主要术语的映射
    term_to_id = {}

    all_terms = []
    for main_label, info in hpo_concepts.items():
        terms = [info['name'][0].lower()] + [t[0].strip().lower() for t in info['synonym']]  # 术语 + 同义词
        all_terms += terms
        term_to_main_label.update({syn: info['name'][0].lower() for syn in terms})  # 记录同义词 → 主术语
        term_to_id.update({syn: main_label for syn in terms})

    index_to_id = {i: term_to_id[all_terms[i]] for i in range(len(all_terms))}
    index_to_term = {i: all_terms[i] for i in range(len(all_terms))}
    json.dump(index_to_id, open(f'{ontology_dict}/index_to_id.json', 'w'))
    json.dump(index_to_term, open(f'{ontology_dict}/index_to_term.json', 'w'))
    json.dump(term_to_main_label, open(f'{ontology_dict}/term_to_main_label.json', 'w'))

    # 🔹 4. 转换为 numpy 数组，并存入 Faiss
    ontology_embeddings = np.array(get_embs(all_terms, tokenizer, model, device=device)).astype('float32')
    faiss.normalize_L2(ontology_embeddings)

    faiss_index = faiss.IndexFlatIP(ontology_embeddings.shape[1])
    faiss_index.add(ontology_embeddings)
    faiss.write_index(faiss_index, f'{ontology_dict}/main.index')

    # 🔹 5. Make index focus on lable.vocab
    ids_filtered = open(f'{ontology_dict}/lable.vocab', 'r').read().strip().split('\n')
    indices_filtered = [index for index, id in index_to_id.items() if id in ids_filtered]
    index_to_id_filtered = {i: index_to_id[index] for i, index in enumerate(indices_filtered)}
    index_to_term_filtered = {i: index_to_term[index] for i, index in enumerate(indices_filtered)}
    json.dump(index_to_id_filtered, open(f'{ontology_dict}/index_to_id_lable.json', 'w'))
    json.dump(index_to_term_filtered, open(f'{ontology_dict}/index_to_term_lable.json', 'w'))

    embs = np.array([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)])
    embs_filtered = np.array([embs[int(i)] for i in indices_filtered])

    faiss_index_filtered = faiss.IndexFlatIP(faiss_index.d)
    faiss_index_filtered.add(embs_filtered)
    faiss.write_index(faiss_index_filtered, f'{ontology_dict}/main_lable.index')

    def list2str(l, delimiter=' | '):
        return delimiter.join([str(i) for i in l])

    def combine_lists(lists):
        new_list = []
        for l in lists:
            new_list.append(l)
        return new_list

    concepts = {}
    aid_map = {}
    for id, concept in tqdm(hpo_concepts.items()):
        id = id.replace(':', '_')
        concepts[id] = {
            "id": id,
            "label": concept['name'][0],
            "definition": concept['def'],
            "synonyms": list2str([s[0] for s in concept['synonym']]),
            "xref": concept.get('xref', []),
            "is_a": list2str([c.replace(':', '_') for c in concept['is_a']])
        }
    json.dump(concepts, open(f'{ontology_dict}/id_to_concept.json', 'w'))

    def get_umls_name(s):
        from dotenv import load_dotenv   # PhenoRAG patch: lazy
        import requests                  # PhenoRAG patch: was never imported upstream
        load_dotenv()
        apikey = os.getenv('UMLS_API_KEY')
        version = 'current'

        base_uri = 'https://uts-ws.nlm.nih.gov'
        path = '/search/' + version
        query = {'string':s, 'apiKey':apikey}
        output = requests.get(base_uri + path, params=query)
        output.encoding = 'utf-8'
            
        outputJson = output.json()
        
        names = []
        if len(outputJson['result']) == 0:
            print('No results found for ' + identifier +'\n')        
        else:  
            names = [r['name'] for r in outputJson['result']['results'] if r['rootSource'] != 'HPO']
        return names

    if umls_synonyms:   # PhenoRAG patch: explicit path, no network fallback
        if not os.path.exists(umls_synonyms):
            raise FileNotFoundError(
                f'UMLS synonym file not found: {umls_synonyms}. Omit --umls_synonyms to build the '
                'index without the "UMLS synonyms" linking field (this changes the linker prompt).')
        umls_syns = json.load(open(umls_synonyms, 'r'))
        concepts = {id: concept | {'UMLS synonyms': umls_syns[id]['UMLS synonyms'] if id in umls_syns else ''} for id, concept in tqdm(concepts.items())}
        # 去除UMLS重复的id
        for id, concept in concepts.items():
            ori_syns = [concept['label'].lower()] + [s.lower() for s in concept['synonyms'].split(' | ')]
            new_umls_syns = [c for c in concept['UMLS synonyms'].split(' | ') if c.lower() not in ori_syns]
            concepts[id]['UMLS synonyms'] = list2str(new_umls_syns)
        json.dump(concepts, open(f'{ontology_dict}/id_to_concept.json', 'w'))

    # def build_lexicons(df):
    #     df = df[(df["LAT"]=="ENG") & df["STR"].notna()].copy()
    #     df["STR"] = df["STR"].str.casefold()
    #     cui2syns = df.groupby("CUI")["STR"].agg(set).to_dict()
    #     df["SAB_CODE"] = df["SAB"] + ":" + df["CODE"]
    #     sabcode2syns = df.groupby("SAB_CODE")["STR"].agg(set).to_dict()
    #     return cui2syns, sabcode2syns

    # def get_xrefs(xrefs, cui2syns, sabcode2syns):
    #     out = set()
    #     for x in xrefs:
    #         sab, code = x.split(":", 1)
    #         if sab == "UMLS":
    #             out |= set(cui2syns.get(code, ()))
    #         else:
    #             out |= set(sabcode2syns.get(f"{sab}:{code}", ()))
    #     return out

    # if ontology_dict == '../dict/HPO':
    #     cols = ["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
    #     usecols = ["CUI", "LAT", "SAB", "CODE", "STR"]
    #     df = pd.read_csv("../data/2024AA/META/MRCONSO.RRF",sep="|",names=cols+["_extra"],dtype=str,usecols=usecols)
    #     cui2syns, sabcode2syns = build_lexicons(df)

    #     concepts = {id: concept | {'cross-reference synonyms': list2str(get_xrefs(concept['xref'], cui2syns, sabcode2syns))} for id, concept in tqdm(concepts.items())}
    #     # 去除UMLS重复的id
    #     for id, concept in concepts.items():
    #         ori_syns = [concept['label'].lower()] + [s.lower() for s in concept['synonyms'].split(' | ')]
    #         new_xref_syns = [c for c in concept['cross-reference synonyms'].split(' | ') if c.lower() not in ori_syns]
    #         concepts[id]['cross-reference synonyms'] = list2str(new_xref_syns)
    #     json.dump(concepts, open(f'{ontology_dict}/id_to_concept.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology_dict', help='folder of the ontology dictionary', type=str, required=True)
    parser.add_argument('--ccr', help='candidate concept retrieval model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    # PhenoRAG patch: --device and --umls_synonyms, see PATCHES.md
    parser.add_argument('--device', help='torch device for the encoder', type=str, default=None)
    parser.add_argument('--umls_synonyms', help="path to AutoPCR's HPO_UMLS-synonyms.json; omitted "
                        "builds the index without the 'UMLS synonyms' linking field", type=str)
    args = parser.parse_args()

    # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    # ../../sapbert/train/tmp/sapbert_umls-hpo
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.ccr)  
    model = AutoModel.from_pretrained(args.ccr).to(device)
    model.eval()

    build_index(args.ontology_dict, tokenizer, model, device=device, umls_synonyms=args.umls_synonyms)