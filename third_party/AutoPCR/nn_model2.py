import time
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import json
from tqdm import tqdm
import re
import yaml
import os
import torch
from nltk.tokenize import word_tokenize
import nltk

from utils import prompt, GROUNDING_PROMPT
from build_dict import lemmatizer, get_wordnet_pos


def get_description(metadata, fields=None):
    if fields is None:
        if isinstance(metadata, dict):
            return yaml.safe_dump({k: v for k, v in metadata.items() if v}, sort_keys=False).strip()
        if isinstance(metadata, str):
            return metadata
    elif isinstance(fields, (list, set, tuple)):
        # if ('definition' in metadata and metadata['definition']) or ('synonyms' in metadata and 'synonyms' in metadata['synonyms']):
        #     fields = ("label", "definition", 'synonyms')
        return yaml.safe_dump({k: v for k, v in metadata.items() if k in fields and v}, sort_keys=False).strip()
    elif isinstance(fields, str) and isinstance(metadata, dict):
        return metadata[fields]


def gen_grounding_prompt2(mention, concepts, text, linking_fields=None, examples=None):
    system_prompt = GROUNDING_PROMPT
    if examples:
        system_prompt += f'Here are some examples:\n' + "\n".join(examples)+ '\n'

    if linking_fields is None:
        linking_fields = ("label", "definition", 'synonyms', 'UMLS synonyms') # 'synonyms'
    
    _candidates = []
    for i, concept in enumerate(concepts):
        _candidates.append(f'ID: {concept["id"]}')
        _candidates.append(get_description(concept, linking_fields) + "\n")

    system_prompt += '\nBelow are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)
    # system_prompt += '\nBelow are the context:\n\n' +  text
    # 'If the chosen concept is broader or narrower than the given entity, return \"None\".' (broader)
    mentions = f'[Entity to link]\n{get_description(mention, fields=["label"])}'
    # mentions += f'\ncontext: {text}'
    # mentions += '\n\nBelow are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)     
    return {"user": mentions, "system": system_prompt}


def gen_grounding_prompt2_llama(mention, concepts, text, linking_fields=None, examples=None):
    system_prompt = GROUNDING_PROMPT
    if examples:
        system_prompt += f'Here are some examples:\n' + "\n".join(examples)+ '\n'

    if linking_fields is None:
        linking_fields = ("label", "definition", 'synonyms', 'UMLS synonyms')
    
    _candidates = []
    real_id = {}
    for i, concept in enumerate(concepts):
        # _candidates.append(f'ID: {concept["id"]}')
        _candidates.append(f'ID: {i+1}')
        real_id[i+1] = concept["id"]
        _candidates.append(get_description(concept, linking_fields) + "\n")

    system_prompt += 'Below are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)
    # system_prompt += '\nBelow are the text:\n\n' +  text
    # 'If the chosen concept is broader or narrower than the given entity, return \"None\".' (broader)
    mentions = f'[Entity to link]\n{get_description(mention, fields=["label"])}'    
    return {"user": mentions, "system": system_prompt, "real_id": real_id}


class bioTag_SapBERT:
    # PhenoRAG patch (see PATCHES.md), three changes to __init__, all keyword-only with upstream
    # defaults preserved:
    #   prompt_fn     -- the entity-linking LLM client. Upstream always called the module-level
    #                    ``utils.llm.prompt`` (OpenAI / Groq / Together / vLLM over the network).
    #                    HCY is patient data and may not be sent to a hosted model, so the baseline
    #                    runs inject a local-model client here instead. Same seam as
    #                    hpo_extraction.baselines.rag_hpo_runner.LocalLlamaClient.
    #   device        -- the SapBERT encoder was hard-wired to .cuda(); it has to be able to sit on
    #                    CPU beside a 70B linker.
    #   use_gpu_index -- the FAISS index was unconditionally copied to GPU 0. It is a 40k x 768
    #                    IndexFlatIP, i.e. milliseconds on CPU, and faiss-gpu==1.7.2 against this
    #                    env's torch/CUDA is an avoidable failure mode. Default off.
    def __init__(self, model_files, ccr, tau_1, tau_2, k, el, api_provider, api_key, seed, use_cache,
                 cand_path='', prompt_fn=None, device=None, use_gpu_index=False):
        self.model_type = 'sapbert'
        # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
        # ../../sapbert/train/tmp/sapbert_umls-hpo
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_fn = prompt_fn or prompt
        self.tokenizer = AutoTokenizer.from_pretrained(ccr)
        self.model = AutoModel.from_pretrained(ccr).to(self.device)
        self.model.eval()
        index = faiss.read_index(model_files['index'])
        if use_gpu_index:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        self.index = index
        self.index_to_id = json.load(open(model_files['index_to_id'], 'r'))
        self.index_to_term = json.load(open(model_files['index_to_term'], 'r'))
        self.id_to_concept = json.load(open(model_files['id_to_concept'], 'r'))
        self.firstword_to_id = json.load(open(model_files['firstword_to_id'], 'r'))
        self.abbr = json.load(open(model_files['abbr'], 'r'))
        self.llm_range = (tau_2, tau_1)
        self.k = k
        self.use_cache = use_cache
        self.cache = {}
        self.el = el
        self.api_provider = api_provider
        self.api_key = api_key
        self.seed = seed
        self.cand_path = cand_path
        self.counter = 0

    def encode(self, all_names, bs=128):
        all_embs = []
        for i in np.arange(0, len(all_names), bs):
            toks = self.tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                            padding="max_length", 
                                            max_length=100, 
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {}
            for k,v in toks.items():
                toks_cuda[k] = v.to(self.device)   # PhenoRAG patch: was v.cuda()
            with torch.no_grad():                  # PhenoRAG patch: encoding is inference-only
                cls_rep = self.model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach().numpy())
        if all_embs:
            all_embs = np.concatenate(all_embs, axis=0)
        return all_embs

    def get_concept(self, index):
        concept = self.id_to_concept[self.index_to_id[index].replace(':', '_')]
        return concept
    
    def get_concept_from_index(self, index):
        concept = self.id_to_concept[self.index_to_id[index].replace(':', '_')].copy()
        term = self.index_to_term[index]
        if concept['label'] == term:
            return concept
        synoyms = concept['synonyms'].split(' | ')
        assert term in synoyms
        new_synonyms = [concept['label']]
        for s in synoyms:
            if s != term:
                new_synonyms.append(s)
        concept['label'] = term
        concept['synonyms'] = ' | '.join(new_synonyms)
        return concept

    def get_cand(self, x, Top_N):
        if not x:
            return {}
        embs = self.encode(x).astype('float32')
        faiss.normalize_L2(embs)
        D, I = self.index.search(embs, Top_N)
        concepts = {}
        for i in range(len(x)):
            concepts[i] = [(self.get_concept(str(I[i][j])), self.index_to_id[str(I[i][j])], float(D[i][j]), str(I[i][j])) for j in range(Top_N)]
        return concepts

    def predict(self, x, Top_N=3):
        concepts = self.get_cand(x, Top_N)
        lines = ''
        for i in range(len(x)):
            lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(Top_N)]) + '\n'
        return lines

    def predict_llm(self, x, texts, confidence=('HIGH')):
        self.counter += len(x)
        concepts = self.get_cand(x, self.k)
        # for i, c in concepts.items():
        #     if x[i] == 'oral anomalies':
        #         print(x[i], [(c[j][0]['label'], c[j][0]['id'], c[j][2]) for j in range(self.k)])
        pr = {}
        for i in range(len(x)):
            if self.use_cache and x[i] in self.cache:
                concepts[i] = self.cache[x[i]]
                continue
            # if ' ' not in x[i]:
            #     continue
            if ' ' not in x[i]:
                # TODO: x[i] may be plural
                if id:=self.abbr.get(x[i].lower(), ''):
                    concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                    continue
                tokens = word_tokenize(x[i].strip().lower().replace('-',' - ').replace('/',' / '))
                if len(tokens) == 1:
                    if tokens[0] in self.firstword_to_id:
                        # print(tokens)
                        id = self.firstword_to_id[tokens[0]][0]
                        concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                        continue
                    token_pos = nltk.pos_tag(tokens)
                    lemmas = [lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])) for token in token_pos]
                    if lemmas[0] in self.firstword_to_id:
                        # print(lemmas)
                        id = self.firstword_to_id[lemmas[0]][0]
                        concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                    continue
            if self.llm_range[0] <= concepts[i][0][2] < self.llm_range[1]:
                # concepts_ = {}
                # for k in range(len(concepts[i])):
                #     if concepts[i][k][0]['id'] not in concepts_:
                #         concepts_[concepts[i][k][0]['id']] = concepts[i][k][0]
                #     if len(concepts_) == self.k:
                #         break
                # concepts_ = list(concepts_.values())
                # key: (concepts[i][k][0]['id'], concepts[i][k][0]['label'])
                concepts_ = list({concepts[i][k][0]['id']: concepts[i][k][0] for k in range(self.k)}.values())
                pr[i] = gen_grounding_prompt2({'label': x[i]}, concepts_, texts[i])
        # print([pr[i] for i in pr if x[i] == 'thumb anomalies'])
        responses = self.prompt_fn(pr, self.el, self.api_provider, self.api_key, self.seed) if self.el != 'none' else {}  # PhenoRAG patch: injected client
        # lines = ''
        anns = []
        for i in range(len(x)):
            if i in responses:
                uri = re.findall('(?<=answer: )[^\n]+', responses[i])
                conf = re.findall('(?<=confidence: )[A-Z]+', responses[i])
                if uri and conf and conf[0] in confidence and 'none' not in uri[0].lower():
                    # lines += f"{uri[0].strip().replace('_', ':')}|{-1.0}" + '\n'
                    # continue
                    id = uri[0].strip()
                    concepts[i] = [(self.id_to_concept.get(id, {}), id.replace('_', ':'), -1.0)] + concepts[i]
            # lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(self.k)]) + '\n'
            anns.append([(concepts[i][j][1], concepts[i][j][2]) for j in range(self.k)])
            if self.use_cache and x[i] not in self.cache:
                self.cache[x[i]] = concepts[i]
        # return lines
        return anns

    # TODO: for local inference
    def predict_llm_batch(self, x_batch, texts_batch, confidence=('HIGH')):
        if os.path.exists(self.cand_path):
            concepts_batch = json.load(open(self.cand_path, 'r'))
            concepts_batch = {int(i): {int(j): concepts[j] for j in concepts} for i, concepts in concepts_batch.items()}
        else:
            concepts_batch = {i: self.get_cand(x, self.k) for i, x in enumerate(x_batch)}
            json.dump(concepts_batch, open(self.cand_path, 'w'))

        if os.path.exists(self.cand_path + '_pr_res'):
            responses_batch = json.load(open(self.cand_path + '_pr_res', 'r'))
            responses_batch = {int(i): {int(j): responses[j] for j in responses} for i, responses in responses_batch.items()}
        else:
            pr_batch = {}
            for j, x in enumerate(x_batch):
                pr, concepts, texts = {}, concepts_batch[j], texts_batch[j]
                for i in range(len(x)):
                    if self.llm_range[0] <= concepts[i][0][2] < self.llm_range[1]:
                        concepts_ = list({concepts[i][k][0]['id']: concepts[i][k][0] for k in range(self.k)}.values())
                        pr[i] = gen_grounding_prompt2_llama({'label': x[i]}, concepts_, texts[i])
                pr_batch[j] = pr
            json.dump(pr_batch, open(self.cand_path + '_pr', 'w'))
            exit(0)
            responses_batch = prompt_batch(self.cand_path + '_pr', self.el)

        lines_batch = []
        for j, x in enumerate(x_batch):
            concepts = concepts_batch[j]
            lines = ''
            for i in range(len(x)):
                if j in responses_batch and i in responses_batch[j]:
                    responses = responses_batch[j]
                    uri = re.findall('(?<=answer: )[^\n]+', responses[i][0])
                    if uri:
                        uri[0] = responses[i][1].get(uri[0].strip(), 'None')
                    conf = re.findall('(?<=confidence: )[A-Z]+', responses[i][0])
                    if uri and conf and conf[0] in confidence and 'none' not in uri[0].lower():
                        lines += f'''{uri[0].replace('_', ':')}|{-1.0}''' + '\n'
                        continue
                lines += '\t'.join([f'{concepts[i][k][1]}|{concepts[i][k][2]}' for k in range(self.k)]) + '\n'
            lines_batch.append(lines)
        return lines_batch


def print_gpu_usage():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
        print(f'GPU memory free: {free / 1024**3} GB')
    else:
        print('CUDA is not available.')